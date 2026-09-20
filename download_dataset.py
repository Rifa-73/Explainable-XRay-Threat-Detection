from roboflow import Roboflow

rf = Roboflow(api_key="dWsqh5MsY7tNZ0vgybAN")

project = rf.workspace("vladutc").project("x-ray-baggage")

version = project.version(3)

dataset = version.download("yolov11")