"""Unit tests for distributed Muon optimizer"""

import math
import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from unittest.mock import MagicMock, Mock, patch

from axolotl.contribs.mit.muon.dist_muon import (
    DistMuon,
    DistMuonOptimizerFactory,
    adjust_lr_rms_norm,
    adjust_lr_spectral_norm,
    zeropower_via_newtonschulz5,
)


class TestDistMuonImportAndInitialization:
    """Test import and basic initialization of DistMuon"""

    def test_dist_muon_import(self):
        """Test that DistMuon can be imported"""
        assert DistMuon is not None
        assert DistMuonOptimizerFactory is not None

    def test_dist_muon_initialization_no_mesh(self):
        """Test DistMuon initialization without device mesh"""
        params = [torch.randn(10, 10, requires_grad=True)]
        optimizer = DistMuon(params, distributed_mesh=None, lr=0.01)

        assert optimizer._device_rank == 0
        assert optimizer._world_size == 1
        assert optimizer._process_group is None

    def test_dist_muon_initialization_with_process_group(self):
        """Test DistMuon initialization with process group"""
        params = [torch.randn(10, 10, requires_grad=True)]

        # Mock process group with proper spec
        from torch.distributed import ProcessGroup
        mock_pg = MagicMock(spec=ProcessGroup)
        with patch("torch.distributed.get_rank", return_value=0):
            with patch("torch.distributed.get_world_size", return_value=4):
                optimizer = DistMuon(params, distributed_mesh=mock_pg, lr=0.01)

                assert optimizer._device_rank == 0
                assert optimizer._world_size == 4
                assert optimizer._process_group == mock_pg

    def test_dist_muon_invalid_lr(self):
        """Test DistMuon raises error with negative learning rate"""
        params = [torch.randn(10, 10, requires_grad=True)]

        with pytest.raises(ValueError, match="invalid learning rate"):
            DistMuon(params, lr=-0.01)

    def test_dist_muon_invalid_mu(self):
        """Test DistMuon raises error with negative momentum"""
        params = [torch.randn(10, 10, requires_grad=True)]

        with pytest.raises(ValueError, match="invalid momentum factor"):
            DistMuon(params, lr=0.01, mu=-0.5)

    def test_dist_muon_invalid_betas(self):
        """Test DistMuon raises error with invalid betas"""
        params = [torch.randn(10, 10, requires_grad=True)]

        with pytest.raises(ValueError, match="invalid betas"):
            DistMuon(params, lr=0.01, betas=(0.9, -0.5))

        with pytest.raises(ValueError, match="invalid betas"):
            DistMuon(params, lr=0.01, betas=(0.9,))

    def test_dist_muon_invalid_adjust_lr(self):
        """Test DistMuon raises error with invalid adjust_lr value"""
        params = [torch.randn(10, 10, requires_grad=True)]

        with pytest.raises(ValueError, match="invalid adjust_lr value"):
            DistMuon(params, lr=0.01, adjust_lr="invalid_value")

    def test_dist_muon_valid_adjust_lr_values(self):
        """Test DistMuon accepts valid adjust_lr values"""
        params = [torch.randn(10, 10, requires_grad=True)]

        # These should not raise
        DistMuon(params, lr=0.01, adjust_lr="spectral_norm")
        DistMuon(params, lr=0.01, adjust_lr="rms_norm")
        DistMuon(params, lr=0.01, adjust_lr=None)


class TestDistMuonDeviceMeshValidation:
    """Test device mesh validation logic"""

    def test_device_mesh_1d_accepted(self):
        """Test that 1D device mesh is accepted"""
        params = [torch.randn(10, 10, requires_grad=True)]

        # Mock 1D device mesh
        mock_mesh = MagicMock(spec=DeviceMesh)
        mock_mesh.ndim = 1
        mock_mesh.size.return_value = 4
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = MagicMock()

        optimizer = DistMuon(params, distributed_mesh=mock_mesh, lr=0.01)
        assert optimizer._world_size == 4
        assert optimizer._device_rank == 0

    def test_device_mesh_2d_rejected(self):
        """Test that 2D+ device mesh raises error"""
        params = [torch.randn(10, 10, requires_grad=True)]

        # Mock 2D device mesh
        mock_mesh = MagicMock(spec=DeviceMesh)
        mock_mesh.ndim = 2

        with pytest.raises(
            ValueError, match="only 1d devicemesh is supported, but got 2d"
        ):
            DistMuon(params, distributed_mesh=mock_mesh, lr=0.01)

    def test_device_mesh_3d_rejected(self):
        """Test that 3D device mesh raises error"""
        params = [torch.randn(10, 10, requires_grad=True)]

        # Mock 3D device mesh
        mock_mesh = MagicMock(spec=DeviceMesh)
        mock_mesh.ndim = 3

        with pytest.raises(
            ValueError, match="only 1d devicemesh is supported, but got 3d"
        ):
            DistMuon(params, distributed_mesh=mock_mesh, lr=0.01)

    def test_invalid_distributed_mesh_type(self):
        """Test that invalid distributed_mesh type raises error"""
        params = [torch.randn(10, 10, requires_grad=True)]

        with pytest.raises(TypeError, match="invalid distributed_mesh type"):
            DistMuon(params, distributed_mesh="invalid", lr=0.01)


class TestDistMuonParameterGrouping:
    """Test parameter grouping logic in DistMuonOptimizerFactory"""

    def test_parameter_grouping_basic(self):
        """Test basic parameter grouping: 2D→Muon, 1D→AdamW"""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),  # Weight: 2D → Muon, Bias: 1D → AdamW
            torch.nn.LayerNorm(128),  # 1D parameters → AdamW
        )

        # Mock training args
        training_args = MagicMock()
        training_args.learning_rate = 0.01

        factory = DistMuonOptimizerFactory()
        optimizer = factory(
            model, training_args, lr=0.01, weight_decay=0.01, device_mesh=None
        )

        # Check that we have both muon and adamw groups
        muon_groups = [g for g in optimizer.param_groups if g["algorithm"] == "muon"]
        adamw_groups = [g for g in optimizer.param_groups if g["algorithm"] == "adamw"]

        assert len(muon_groups) > 0, "Should have muon parameter groups"
        assert len(adamw_groups) > 0, "Should have adamw parameter groups"

        # Check that 2D params are in muon groups
        muon_params = sum(len(g["params"]) for g in muon_groups)
        assert muon_params >= 1, "Linear weight should be in muon group"

        # Check that 1D params are in adamw groups
        adamw_params = sum(len(g["params"]) for g in adamw_groups)
        assert adamw_params >= 2, "Bias and LayerNorm params should be in adamw groups"

    def test_embedding_uses_adamw(self):
        """Test that embedding parameters use AdamW algorithm"""
        # Create model with embedding
        model = torch.nn.Sequential(
            torch.nn.Embedding(1000, 128), torch.nn.Linear(128, 64)
        )

        training_args = MagicMock()
        training_args.learning_rate = 0.01

        factory = DistMuonOptimizerFactory()
        optimizer = factory(
            model, training_args, lr=0.01, weight_decay=0.01, device_mesh=None
        )

        # Find embedding params
        adamw_groups = [g for g in optimizer.param_groups if g["algorithm"] == "adamw"]

        # Embedding should be in adamw groups
        adamw_params = sum(len(g["params"]) for g in adamw_groups)
        assert adamw_params >= 1, "Embedding should be in adamw group"

    def test_weight_decay_grouping(self):
        """Test that weight decay is correctly applied to different groups"""
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
        )

        training_args = MagicMock()
        training_args.learning_rate = 0.01

        factory = DistMuonOptimizerFactory()
        optimizer = factory(
            model, training_args, lr=0.01, weight_decay=0.01, device_mesh=None
        )

        # Check that we have both weight_decay and no_weight_decay groups
        has_decay = any(
            g["weight_decay"] > 0 for g in optimizer.param_groups if g["algorithm"] == "muon"
        )
        has_no_decay = any(
            g["weight_decay"] == 0 for g in optimizer.param_groups
        )

        assert has_decay, "Should have parameter groups with weight decay"
        assert has_no_decay, "Should have parameter groups without weight decay"


class TestNewtonSchulzFunction:
    """Test Newton-Schulz orthogonalization function"""

    def test_newton_schulz_produces_orthogonal_matrix(self):
        """Test that Newton-Schulz produces approximately orthogonal matrices"""
        # Create a random matrix
        X = torch.randn(32, 32, dtype=torch.bfloat16)

        # Apply Newton-Schulz
        U = zeropower_via_newtonschulz5(X)

        # Check that U @ U.T ≈ I
        identity = U @ U.mT
        expected_identity = torch.eye(32, dtype=torch.bfloat16)

        # Allow some tolerance due to bfloat16 precision
        diff = (identity - expected_identity).abs().max().item()
        assert diff < 0.1, f"Matrix should be approximately orthogonal, but max diff is {diff}"

    def test_newton_schulz_output_shape(self):
        """Test that Newton-Schulz preserves shape"""
        shapes = [(32, 32), (64, 32), (32, 64), (128, 64)]

        for shape in shapes:
            X = torch.randn(shape, dtype=torch.bfloat16)
            U = zeropower_via_newtonschulz5(X)
            assert U.shape == X.shape, f"Output shape {U.shape} should match input shape {X.shape}"

    def test_newton_schulz_with_epsilon(self):
        """Test Newton-Schulz with different epsilon values"""
        X = torch.randn(32, 32, dtype=torch.bfloat16)

        # Should not crash with different epsilon values
        U1 = zeropower_via_newtonschulz5(X, epsilon=1e-7)
        U2 = zeropower_via_newtonschulz5(X, epsilon=1e-5)

        assert U1.shape == X.shape
        assert U2.shape == X.shape
        assert not torch.isnan(U1).any(), "Output should not contain NaN"
        assert not torch.isnan(U2).any(), "Output should not contain NaN"


class TestLearningRateAdjustment:
    """Test learning rate adjustment functions"""

    def test_adjust_lr_spectral_norm(self):
        """Test spectral norm learning rate adjustment"""
        lr = 0.01
        param_shape = (128, 64)  # fan_out=128, fan_in=64
        flatten = False

        adjusted_lr = adjust_lr_spectral_norm(lr, param_shape, flatten)

        # Formula: lr * sqrt(fan_out / fan_in)
        expected = lr * math.sqrt(128 / 64)
        assert abs(adjusted_lr - expected) < 1e-6, f"Expected {expected}, got {adjusted_lr}"

    def test_adjust_lr_rms_norm(self):
        """Test RMS norm learning rate adjustment"""
        lr = 0.01
        param_shape = (128, 64)  # fan_out=128, fan_in=64
        flatten = False

        adjusted_lr = adjust_lr_rms_norm(lr, param_shape, flatten)

        # Formula: lr * 0.2 * sqrt(max(fan_out, fan_in))
        expected = lr * 0.2 * math.sqrt(max(128, 64))
        assert abs(adjusted_lr - expected) < 1e-6, f"Expected {expected}, got {adjusted_lr}"

    def test_adjust_lr_with_flatten(self):
        """Test learning rate adjustment with flatten=True"""
        lr = 0.01
        param_shape = (64, 32, 32)  # 3D tensor: fan_out=64, fan_in=32*32=1024
        flatten = True

        adjusted_lr_spectral = adjust_lr_spectral_norm(lr, param_shape, flatten)
        adjusted_lr_rms = adjust_lr_rms_norm(lr, param_shape, flatten)

        # With flatten: fan_out=64, fan_in=1024
        expected_spectral = lr * math.sqrt(64 / 1024)
        expected_rms = lr * 0.2 * math.sqrt(max(64, 1024))

        assert abs(adjusted_lr_spectral - expected_spectral) < 1e-6
        assert abs(adjusted_lr_rms - expected_rms) < 1e-6

    def test_adjust_lr_various_shapes(self):
        """Test learning rate adjustment with various parameter shapes"""
        lr = 0.01
        shapes = [(256, 256), (512, 128), (64, 512), (1024, 1024)]

        for shape in shapes:
            # Should not crash
            lr_spectral = adjust_lr_spectral_norm(lr, shape, flatten=False)
            lr_rms = adjust_lr_rms_norm(lr, shape, flatten=False)

            assert lr_spectral > 0, "Adjusted LR should be positive"
            assert lr_rms > 0, "Adjusted LR should be positive"
            assert not math.isnan(lr_spectral), "Adjusted LR should not be NaN"
            assert not math.isnan(lr_rms), "Adjusted LR should not be NaN"


class TestDistMuonStateInitialization:
    """Test optimizer state initialization"""

    def test_state_lazy_initialization(self):
        """Test that state is lazily initialized on first step"""
        params = [torch.randn(10, 10, requires_grad=True)]
        optimizer = DistMuon(params, lr=0.01)

        # State should be empty before first step
        assert len(optimizer.state) == 0

        # Create gradient and step
        params[0].grad = torch.randn(10, 10)
        optimizer.step()

        # State should now be initialized
        assert len(optimizer.state) > 0
        state = optimizer.state[params[0]]
        assert "momentum" in state
        assert state["momentum"].shape == params[0].shape

    def test_state_muon_no_variance(self):
        """Test that muon algorithm doesn't create variance state"""
        params = [torch.randn(10, 10, requires_grad=True)]
        optimizer = DistMuon(params, lr=0.01)

        params[0].grad = torch.randn(10, 10)
        optimizer.step()

        state = optimizer.state[params[0]]
        assert "momentum" in state
        assert "variance" not in state, "Muon should not have variance state"

    def test_state_adamw_has_variance(self):
        """Test that adamw algorithm creates variance state"""
        # Create 1D parameter (will use adamw)
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        bias = model[0].bias  # 1D parameter

        training_args = MagicMock()
        training_args.learning_rate = 0.01

        factory = DistMuonOptimizerFactory()
        optimizer = factory(model, training_args, lr=0.01, weight_decay=0.0)

        # Create gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        optimizer.step()

        # Check bias state (should use adamw)
        bias_state = optimizer.state[bias]
        assert "momentum" in bias_state
        assert "variance" in bias_state, "AdamW should have variance state"


class TestDistMuonFactoryDeviceMeshExtraction:
    """Test device mesh extraction in DistMuonOptimizerFactory"""

    def test_factory_extracts_dp_shard_dimension(self):
        """Test that factory extracts dp_shard dimension from device mesh"""
        model = torch.nn.Linear(10, 10)
        training_args = MagicMock()
        training_args.learning_rate = 0.01

        # Mock device mesh with dp_shard dimension
        mock_mesh = MagicMock(spec=DeviceMesh)
        mock_mesh.ndim = 1
        mock_mesh.mesh_dim_names = ["dp_shard"]
        mock_mesh.__getitem__.return_value = mock_mesh  # mesh["dp_shard"] returns itself
        mock_mesh.size.return_value = 4
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = MagicMock()

        factory = DistMuonOptimizerFactory()
        optimizer = factory(model, training_args, lr=0.01, device_mesh=mock_mesh)

        # Should have extracted the mesh
        assert optimizer._distributed_mesh is not None
        assert optimizer._world_size == 4

    def test_factory_handles_1d_unnamed_mesh(self):
        """Test that factory handles 1D mesh without named dimensions"""
        model = torch.nn.Linear(10, 10)
        training_args = MagicMock()
        training_args.learning_rate = 0.01

        # Mock 1D device mesh without named dimensions
        mock_mesh = MagicMock(spec=DeviceMesh)
        mock_mesh.ndim = 1
        mock_mesh.mesh_dim_names = []
        mock_mesh.size.return_value = 4
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = MagicMock()

        factory = DistMuonOptimizerFactory()
        optimizer = factory(model, training_args, lr=0.01, device_mesh=mock_mesh)

        # Should use the 1D mesh directly
        assert optimizer._distributed_mesh is not None
        assert optimizer._world_size == 4

    def test_factory_handles_no_device_mesh(self):
        """Test that factory handles device_mesh=None"""
        model = torch.nn.Linear(10, 10)
        training_args = MagicMock()
        training_args.learning_rate = 0.01

        factory = DistMuonOptimizerFactory()
        optimizer = factory(model, training_args, lr=0.01, device_mesh=None)

        # Should work without device mesh
        assert optimizer._distributed_mesh is None
        assert optimizer._world_size == 1
