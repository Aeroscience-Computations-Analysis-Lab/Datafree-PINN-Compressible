import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt    
from src import PolygonBoundaryPoints, FNN, PINN

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=["SST", "LST"])
    args = parser.parse_args()

    if args.case == "SST":
        u_initial =  [1.0, 1.0, 0.0, 0.125, 0.1, 0.0]
        Xmin, Xmax = 0.0, 1.0
        Tmin, Tmax = 0.0, 0.2

    elif args.case == "LST":
        u_initial = [0.445, 3.528, 0.689, 0.5, 0.571, 0.0] 
        Xmin, Xmax = 0.0, 1.0
        Tmin, Tmax = 0.0, 0.14
        
    print(f"Case: {args.case}")
    print(f"u_initial = {u_initial}")
    print(f"X range = [{Xmin}, {Xmax}]")
    print(f"T range = [{Tmin}, {Tmax}]")
    print(f"Device: {device}")
    
    

    vertices = [(Xmin, Tmin), (Xmax, Tmin), (Xmax, Tmax), (Xmin, Tmax)]  
    polygon = PolygonBoundaryPoints(vertices, num_boundary_points=2000)

    boundary_points, edge_points_list = polygon.generate_points_on_edges()
    interior_points = polygon.generate_random_points_inside(81920)
    lbfgs_interior_points = polygon.generate_random_points_inside(40000)

    inputs = torch.tensor(interior_points, dtype=torch.float32, device=device)
    
    layers=[2]+5*[192]+[3]
    model = FNN(inputs, layers, init_type='xavier')
    model.to(device)
    layers2 = [2]+5*[96]+[1]
    model2 = FNN(inputs, layers2, init_type='xavier', output2='non_exp')
    model2.to(device)
    
    x_boundary = edge_points_list[0]
    x_boundary = torch.tensor(x_boundary, dtype=torch.float32, device=device)
    
    u_initial = torch.tensor(u_initial, dtype=torch.float32, device=device)
    
    lbfgs_x_train = torch.tensor(lbfgs_interior_points, dtype=torch.float32, device=device)

    pinn = PINN(model,model2, device=device, Xmin=Xmin, Xmax=Xmax, Tmin=Tmin, Tmax=Tmax)
    pinn.train(x_physics=inputs, x_boundary=x_boundary, u_boundary=u_initial, epochs=2001)
    pinn.train_lbfgs(x_physics=lbfgs_x_train, x_boundary=x_boundary, u_boundary=u_initial, epochs=401)
    
if __name__ == "__main__":
    main()
