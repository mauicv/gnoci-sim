import mujoco
import os
import matplotlib.pyplot as plt

package_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(package_dir, 'src', 'desc', 'gnoci.xml')
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


with mujoco.Renderer(model) as renderer:
    mujoco.mj_step(model, data)
    renderer.update_scene(data, camera='track')
    pixels = renderer.render()
    plt.imshow(pixels)
    plt.show()