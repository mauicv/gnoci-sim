import mujoco
model = mujoco.MjModel.from_xml_path("onshape_export/scene.xml")  # or robot.xml
data = mujoco.MjData(model)
for _ in range(1000):
    mujoco.mj_step(model, data)
