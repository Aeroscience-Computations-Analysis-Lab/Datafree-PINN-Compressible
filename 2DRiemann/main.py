import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt   
from src import FNN, PINN, VolumeGenerator

def main():  
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=["CONFIG1", "CONFIG2", "CONFIG3"])
    args = parser.parse_args()

    if args.case == "CONFIG1":
        ic = torch.tensor([[1.0, -0.75, 0.5, 1.0],[2,0.75,0.5,1.0],[3,-0.75,-0.5,1],[1,0.75,-0.5,1]])
        Xmin, Xmax = 0.0, 1.0
        Ymin, Ymax = 0.0, 1.0
        Tmin, Tmax = 0.0, 0.4

    elif args.case == "CONFIG2":
        ic = torch.tensor([[0.5313,0.0,0.0,0.4],[1.0,0.7276,0,1.0],[0.8,0.0,0.0,1.0],[1.0, 0, 0.7276, 1.0]]) 
        Xmin, Xmax = 0.0, 1.0
        Ymin, Ymax = 0.0, 1.0
        Tmin, Tmax = 0.0, 0.25

    elif args.case == "CONFIG3":
        ic = torch.tensor([[0.5313,0.0,0.0,0.4],[1.0,0.7276,0,1.0],[0.8,0.0,0.0,1.0],[1.0, 0, 0.7276, 1.0]])
        Xmin, Xmax = 0.0, 1.0
        Ymin, Ymax = 0.0, 1.0
        Tmin, Tmax = 0.0, 0.25
        
    print(f"Case: {args.case}")
    print(f"ic = {ic}")
    print(f"X range = [{Xmin}, {Xmax}]")
    print(f"Y range = [{Ymin}, {Ymax}]")
    print(f"T range = [{Tmin}, {Tmax}]")
    print(f"Device: {device}")

    volume_gen = VolumeGenerator(Xmin, Xmax, Ymin, Ymax, Tmin, Tmax)
    x_train = volume_gen.generate_volume_points(100000)
    x_train_lbfgs = volume_gen.generate_volume_points(40000)
    
    inputs_lbfgs = x_train_lbfgs.to(dtype=torch.float32, device=device)
    inputs = x_train.to(dtype=torch.float32, device=device)
    layers=[3]+5*[96]+[4]

    model = FNN(inputs, layers, init_type='xavier')
    model.to(device)
    
    layers2 = [3] + 5 * [96] + [1]
    model2 = FNN(inputs, layers2, init_type='xavier', output2='not_exp') 
    model2.to(device)
    
    x_initial = volume_gen.generate_surface_points(surface='Tmin', num_points=10000)
    x_initial = x_initial.to(dtype=torch.float32, device=device)
    ic = ic.to(dtype=torch.float32, device=device)
    
    pinn = PINN(model, model2, device=device)
    # pinn.to(device) 
    pinn.train(inputs, x_initial, ic, epochs=11)
    pinn.train_lbfgs(inputs_lbfgs, x_initial, ic, epochs=1001)
    
if __name__ == "__main__":
    main()
