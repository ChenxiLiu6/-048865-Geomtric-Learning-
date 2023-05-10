# 048865-Geomtric-Learning
## Geometric Moments and Neural Shape Analysis 

1. This project implemented a deep neural network which can be applied to point clouds and processes on the points’ coordinates for classification task, which is known as the classic **PointNet**.

2. In order to better capture local structures, this project also implemented a deep neural network architecture **Momenet** according to paper: _"Momenet: Flavor the Moments in Learning to Classify Shapes"_, which takes the geometry context in the form of geometric moments of 3D shapes into consideration and adds polynomial functions to the origin point cloud coordinates.

3. The main models that have been built in this project include:
(1) Classic PointNet: **pointnet** 
(2) Classic Momenet (1st and 2nd order moments): **momenet**
(3) 1st, 2nd, and 3rd order moments: **momenet3**
(4) PointNet with Vertex Normals: **pointnet_vn**
(5) Momenet with Vertex Normals: **momenet_vn2**
(6) 1st, 2nd, and 3rd Momenet with Vertex Normals: **momenet_vn3**
(7) PointNet with Harmonic Pre-lifting: **pointnet_hp**
