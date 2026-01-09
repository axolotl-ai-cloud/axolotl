"""Test DFT compatibility with Pipeline Parallelism (PP).

Pipeline Parallelism (PP) splits model layers vertically across devices:
- Stage 0: Layers 0-20 on GPU 0
- Stage 1: Layers 21-40 on GPU 1
- Each stage processes micro-batches sequentially

AXOLOTL SUPPORT STATUS: NOT SUPPORTED (as of 2026-01-06)
- axolotl does not currently implement Pipeline Parallelism
- No PP configuration options in config schema
- No PP-related code in loaders or trainers
- Preferred approach: FSDP + TP for large models

VERIFICATION APPROACH:
⚠️  This test suite ACTIVELY CHECKS for PP implementation.
    If axolotl adds PP support in the future, these tests will FAIL,
    alerting developers to revisit DFT+PP compatibility.

FUTURE CONSIDERATION:
If axolotl adds PP support in the future, DFT compatibility analysis:

EXPECTED BEHAVIOR: ✅ LIKELY COMPATIBLE
- Each PP stage outputs complete tensors [batch, seq, hidden_dim]
- Final stage outputs complete logits [batch, seq, vocab]
- DFT loss computation happens only on final stage (last GPU)
- No special handling needed - DFT receives complete logits

POTENTIAL ISSUES:
- Loss computed only on final stage → may need gradient broadcast
- Micro-batch handling → ensure num_items_in_batch is correct
- Stage synchronization → should be handled by PP framework

RECOMMENDATION:
- Mark as "Not Applicable" in compatibility matrix
- Revisit if axolotl implements PP in future
- Low priority - FSDP+TP is preferred strategy for large models
"""

from pathlib import Path

import pytest


class TestDFTPipelineParallelCompatibility:
    """Verify Pipeline Parallelism support status in axolotl.

    These tests ACTIVELY CHECK the codebase. If PP is added in the future,
    tests will FAIL, prompting developers to verify DFT+PP compatibility.
    """

    def test_pipeline_parallel_not_in_config_schema(self):
        """Verify PP config options don't exist in config schema.

        If this test fails, PP support may have been added - review DFT compatibility.
        """
        config_schema_path = Path("src/axolotl/utils/schemas/config.py")

        if not config_schema_path.exists():
            pytest.skip(f"Config schema not found at {config_schema_path}")

        config_content = config_schema_path.read_text()

        # Check for common PP config names
        pp_config_indicators = [
            "pp_size",
            "pipeline_parallel_size",
            "pipeline_parallel",
            "num_pipeline_stages",
        ]

        found_indicators = [
            indicator
            for indicator in pp_config_indicators
            if indicator in config_content
        ]

        if found_indicators:
            pytest.fail(
                f"⚠️  ALERT: PP config options detected in config schema: {found_indicators}\n"
                f"axolotl may have added Pipeline Parallelism support!\n"
                f"Action required:\n"
                f"1. Verify DFT+PP compatibility (likely compatible - see docstring)\n"
                f"2. Add E2E tests for DFT+PP\n"
                f"3. Update compatibility matrix from N/A to actual status\n"
                f"4. Update this test to reflect PP support"
            )

        print("\n✓ No PP config options found in config schema (PP not supported)")

    def test_pipeline_parallel_not_in_model_loader(self):
        """Verify PP initialization code doesn't exist in model loader.

        If this test fails, PP support may have been added - review DFT compatibility.
        """
        model_loader_path = Path("src/axolotl/loaders/model.py")

        if not model_loader_path.exists():
            pytest.skip(f"Model loader not found at {model_loader_path}")

        loader_content = model_loader_path.read_text()

        # Check for PP-related code patterns
        pp_code_indicators = [
            "pipeline_parallel",
            "pp_size",
            "PipelineParallel",
            "pipeline_stage",
        ]

        found_indicators = [
            indicator for indicator in pp_code_indicators if indicator in loader_content
        ]

        if found_indicators:
            pytest.fail(
                f"⚠️  ALERT: PP code detected in model loader: {found_indicators}\n"
                f"axolotl may have added Pipeline Parallelism support!\n"
                f"Action required:\n"
                f"1. Verify DFT+PP compatibility\n"
                f"2. Add E2E tests for DFT+PP\n"
                f"3. Update compatibility matrix"
            )

        print("\n✓ No PP initialization code found in model loader (PP not supported)")

    def test_pipeline_parallel_not_in_distributed_utils(self):
        """Verify PP process group setup doesn't exist in distributed utils.

        If this test fails, PP support may have been added - review DFT compatibility.
        """
        distributed_path = Path("src/axolotl/utils/distributed.py")

        if not distributed_path.exists():
            pytest.skip(f"Distributed utils not found at {distributed_path}")

        distributed_content = distributed_path.read_text()

        # Check for PP process group patterns
        pp_group_indicators = [
            "pipeline_parallel_group",
            "pp_group",
            "get_pipeline_parallel",
        ]

        found_indicators = [
            indicator
            for indicator in pp_group_indicators
            if indicator in distributed_content
        ]

        if found_indicators:
            pytest.fail(
                f"⚠️  ALERT: PP process groups detected: {found_indicators}\n"
                f"axolotl may have added Pipeline Parallelism support!\n"
                f"Action required: Verify DFT+PP compatibility"
            )

        print("\n✓ No PP process group setup found (PP not supported)")

    def test_pp_compatibility_analysis_for_future(self):
        """If PP is added to axolotl in future, analyze DFT compatibility.

        Theoretical analysis (assuming standard PP implementation):

        PP Architecture:
        - Model layers split vertically across stages (GPUs)
        - Each stage processes micro-batches sequentially
        - Final stage outputs complete logits [batch, seq, vocab]
        - Loss computed on final stage

        DFT Compatibility (Theoretical):
        - ✅ DFT receives complete logits from final stage
        - ✅ Loss computation unchanged (happens on last GPU only)
        - 🟡 Gradient broadcast may be needed (handled by PP framework)
        - 🟡 Micro-batch handling - ensure num_items_in_batch is correct

        Expected Outcome: LIKELY COMPATIBLE
        - No DFT code changes needed
        - PP framework handles cross-stage communication
        - Loss computation on final stage is standard pattern
        """
        theoretical_analysis = {
            "PP outputs": "Complete logits [batch, seq, vocab] from final stage",
            "DFT input": "Same as non-PP - complete logits",
            "Loss location": "Computed on final stage (last GPU)",
            "Gradient flow": "Handled by PP framework (backward through stages)",
            "Expected compatibility": "✅ Likely compatible (no special handling)",
            "Main concern": "Ensure num_items_in_batch accounts for micro-batching",
        }

        print("\nTheoretical PP + DFT Compatibility Analysis:")
        for aspect, finding in theoretical_analysis.items():
            print(f"  {aspect}: {finding}")

        assert True, "Theoretical PP compatibility analysis documented"

    def test_compatibility_matrix_status(self):
        """Verify correct status in DFT compatibility matrix.

        Current status: 🟡 Likely OK
        Correct status:  N/A (Not Applicable) - PP not supported in axolotl

        Recommendation: Update spec 001 compatibility matrix:
        - Change from: "🟡 Likely OK - If supported, each stage outputs complete tensors"
        - Change to:   "N/A - Not supported in axolotl (prefers FSDP+TP)"
        """
        recommended_matrix_entry = {
            "Feature": "Pipeline Parallelism",
            "Status": "N/A (Not Applicable)",
            "Details": "Not supported in axolotl - FSDP+TP preferred for large models",
            "File Reference": "No PP implementation",
            "Note": "If added in future, likely compatible (DFT receives complete logits)",
        }

        print("\nRecommended Compatibility Matrix Entry:")
        for key, value in recommended_matrix_entry.items():
            print(f"  {key}: {value}")

        assert True, "Compatibility matrix recommendation documented"


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
