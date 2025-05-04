import trimesh

mesh = trimesh.load("/Users/utkarshsingh/Downloads/ModelNet40/ModelNet40/car/train/car_0001.off")
mesh.export("data/meshes/car_0001.stl")
