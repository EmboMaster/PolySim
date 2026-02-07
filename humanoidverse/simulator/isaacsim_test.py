import os
import sys
import time
import argparse
from datetime import timedelta

# ---------------------------------------------------------
# 1. Setup Environment Variables
# ---------------------------------------------------------
print("[Warmup] Setting environment variables...")
os.environ["NV_SHADER_CACHE_SIZE"] = "1000000000" 
os.environ["DISPLAY"] = ":0" 

try:
    # ---------------------------------------------------------
    # 2. Start AppLauncher
    # ---------------------------------------------------------
    from omni.isaac.lab.app import AppLauncher
    
    parser = argparse.ArgumentParser(description="Isaac Sim First Run Warmup")
    AppLauncher.add_app_launcher_args(parser)
    
    print("[Warmup] Starting AppLauncher (Headless Mode)...")
    args_cli = parser.parse_args(["--headless"])
    
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    print(f"[Warmup] SimulationApp started successfully!")

    # ---------------------------------------------------------
    # 3. Import Core Libraries
    # ---------------------------------------------------------
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni.isaac.core.utils.viewports import set_camera_view
    
    # ---------------------------------------------------------
    # 4. Build Test Scene
    # ---------------------------------------------------------
    print("[Warmup] Building test scene (Ground Plane + Light)...")
    
    sim = SimulationContext()
    
    # Create Ground
    prim_utils.create_prim("/World/Ground", "Cube", position=(0, 0, -0.5), scale=(100, 100, 1))
    
    # [FIXED HERE] Create Light with 'inputs:intensity' for newer USD versions
    prim_utils.create_prim(
        "/World/Light", 
        "DistantLight", 
        attributes={"inputs:intensity": 1000.0}
    )

    # Set Camera
    set_camera_view([10, 10, 10], [0, 0, 0])

    # ---------------------------------------------------------
    # 5. Execute Reset and Shader Compilation
    # ---------------------------------------------------------
    print("="*60)
    print("[WARNING] Starting Shader compilation and Physics initialization.")
    print("[WARNING] This may take 5 to 20 minutes. CPU usage may MAX OUT.")
    print("[WARNING] Please DO NOT close the program. Wait patiently.")
    print("="*60)
    
    start_time = time.time()
    
    print("[Warmup] Executing sim.reset() ...")
    sim.reset()
    print(f"[Warmup] Reset finished! Time taken: {str(timedelta(seconds=int(time.time() - start_time)))}")

    # ---------------------------------------------------------
    # 6. Physics Warmup Steps
    # ---------------------------------------------------------
    print("[Warmup] Executing 100 Physics Warmup Steps...")
    
    for i in range(100):
        sim.step(render=False)
        if i % 10 == 0:
            print(f"   - Step {i}/100 done...")
    
    print("[Warmup] Physics warmup finished.")

    # ---------------------------------------------------------
    # 7. Clean up
    # ---------------------------------------------------------
    print("="*60)
    print("[SUCCESS] Isaac Sim initialization complete.")
    print("="*60)
    
    simulation_app.close()

except Exception as e:
    import traceback
    print("\n[ERROR] An error occurred:")
    traceback.print_exc()
    sys.exit(1)