"""Print the pod environment relevant to the sonicmoe NVFP4 kernel work."""

import _common  # noqa: F401  (sys.path bootstrap)


def main():
    import torch

    print(f"torch                {torch.__version__}")
    print(f"cuda available       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device               {torch.cuda.get_device_name()}")
        cap = torch.cuda.get_device_capability()
        print(f"capability           sm_{cap[0]}{cap[1]}")
    print(f"float4_e2m1fn_x2     {hasattr(torch, 'float4_e2m1fn_x2')}")
    print(f"float8_e4m3fn        {hasattr(torch, 'float8_e4m3fn')}")

    try:
        import quack

        print(f"quack                {getattr(quack, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"quack                MISSING ({e})")
    try:
        import cutlass

        print(f"nvidia-cutlass-dsl   {getattr(cutlass, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"nvidia-cutlass-dsl   MISSING ({e})")

    from fp4_cute import fp4_cute_available

    print(f"fp4_cute_available   {fp4_cute_available()}")


if __name__ == "__main__":
    main()
