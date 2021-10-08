
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 1000



def preprocess(dataframe):
    df=dataframe
    rows=df['Innings']==1
    df=df[rows]
    
    
    overs_completed = df['Over'].to_numpy()
    total_overs = df['Total.Overs'].to_numpy()
    
    overs_remaining = total_overs - overs_completed
    innings_total_score = df['Innings.Total.Runs'].to_numpy()
    current_score = df['Total.Runs'].to_numpy()
    runs_remaining = innings_total_score - current_score
    wickets_remaining = df['Wickets.in.Hand'].to_numpy()
    
    
    return overs_remaining , runs_remaining , wickets_remaining 



def squared_error_loss_20(params, args):
    Z,b,func = params[0],params[1],params[2]
    over = args[0]
    runs = args[1]
    predicted_run=func_1(Z,b,over)
    loss=np.sum((predicted_run - runs)**2)
    return loss    


def squared_error_loss_11(param, arg):
    L = param[10]
    # print(L)
    
    overs   = arg[0]
    runs    = arg[1]
    wickets = arg[2]
    loss=0
    for i in range(len(wickets)):
        if (runs [i] > 0 and wickets[i]>0):
            pred = func_2(param[wickets[i]-1],L,overs[i])
            loss+=(pred - runs[i])**2
    # print(count)        
    return loss



def DuckworthLewis20Params(path):
    df=pd.read_csv(path)
    overs_remaining , runs_remaining , wickets_remaining=preprocess(df)
    initial_parameters = [10.0, 20.0, 35.0, 50.0, 70.0, 100.0, 140.0, 180.0, 235.0,280.0] 
    parameters_B = list(0.3*np.ones((10,1)))
    opt_Z=[]
    opt_b=[]
    
    err=[]
    for i in range(1,11):
        
        run_remain=runs_remaining[wickets_remaining==i]
        over_remain=overs_remaining[wickets_remaining==i]
        parameters = minimize(squared_error_loss_20,[initial_parameters[i-1],parameters_B[i-1],0],
                      args=[over_remain,
                            run_remain
                            ],
                      method='L-BFGS-B')
        optimum_params, squared_error = parameters['x'], parameters['fun']
        
        err.append(squared_error)
        opt_Z.append(optimum_params[0])
        opt_b.append(optimum_params[1])
    
    for i in range(10):
        print ('Parameter Z' + str(i+1) + ' :: ' + str(opt_Z[i]))    
    for i in range(10):
        print ('Parameter b' + str(i+1) + ' :: ' + str(opt_b[i]))
    
    print ('Error per point :: ' + str(sum(err)/len(overs_remaining[wickets_remaining!=0])))    
    
    plot_func_1(opt_Z,opt_b)
    return opt_Z,opt_b


def DuckworthLewis11Params(path):
    
    df=pd.read_csv(path)
    overs_remaining , runs_remaining , wickets_remaining=preprocess(df)
    initial_parameters = [10.0, 20.0, 35.0, 50.0, 70.0, 100.0, 140.0, 180.0, 235.0,280.0,0.5] #Random Values
    opt_Z=[]
    opt_L=0
    err=[]
    # print(overs_remaining )
    # print(runs_remaining)
    # print(wickets_remaining)
    parameters = minimize(squared_error_loss_11,[initial_parameters],
                      args=[overs_remaining ,
                            runs_remaining,
                            wickets_remaining 
                            ],
                      method='L-BFGS-B')
    optimum_params, squared_error = parameters['x'], parameters['fun']
    err.append(squared_error)
    opt_Z.append(optimum_params[:10])
    opt_L=optimum_params[10]
    
    
    # print ('Parameter L :: ' + str(optimized_params[10]))
    for i in range(10):
        print ('Parameter Z' + str(i+1) + ' :: ' + str(opt_Z[0][i]))    
    # for i in range(10):
    print ('Parameter L'  + ' :: ' + str(opt_L))
    print ('Error per point :: ' + str(sum(err)/len(overs_remaining[wickets_remaining!=0]))) 
    
    plot_func_2(list(opt_Z[0]),opt_L)
    return opt_Z,opt_L

def func_1(z, b, u):
    return z * (1 - np.exp(-b*u))

def func_2(z, l, u):
    return z * (1 - np.exp(-l*u/z))


def plot_func_1(Z,b):

    plt.figure(figsize=(10,7)) #Fig Size
    plt.xlabel('Overs remaining (u)')
    plt.ylabel('Percentage of resource remaining')
    plt.xlim((0, 50))
    plt.ylim((0, 100))
    plt.xticks(list(np.linspace(0,50,num=11)))
    plt.yticks(list(np.linspace(0,100,num=11)))
    max_resource = func_1(Z[9],b[9], 50)
    overs = np.linspace(0, 50, num=51)
    line_cord=[]
    
    for i in range(len(overs)):
        line_cord.append(2*i)
    plt.plot(overs, line_cord, color='blue')
    
    #Plot Resources Remaining vs overs Remaining
    for i in range(10):
        fraction= func_1(Z[i],b[i], overs)/max_resource
        plt.plot(overs,100*fraction, label='Z['+str(i+1)+']')
        plt.legend()
    plt.grid()
  


def plot_func_2(Z,L):

    plt.figure(figsize=(10,7)) #Fig Size
    plt.xlabel('Overs remaining (u)')
    plt.ylabel('Percentage of resource remaining')
    plt.xlim((0, 50))
    plt.ylim((0, 100))
    plt.xticks(list(np.linspace(0,50,num=11)))
    plt.yticks(list(np.linspace(0,100,num=11)))
    max_resource = func_2(Z[9],L, 50)
    overs = np.linspace(0, 50, num=51)
    line_cord=[]
   
    for i in range(len(overs)):
       line_cord.append(2*i)
    plt.plot(overs, line_cord, color='blue')
    for i in range(10):
        fraction= func_2(Z[i],L, overs)/max_resource
        plt.plot(overs,100*fraction, label='Z['+str(i+1)+']')
        plt.legend()
    
    plt.grid()




     
  
def main():
    path=r"D:\Data Analytics\Assignment2\04_cricket_1999to2011_2.csv"
    # Z,b=DuckworthLewis20Params(path)
    
    # Slope at u=0 for first function
    # mult=np.multiply(np.array(Z),np.array(b))
    # print(mult)
    
    
    Z,l=DuckworthLewis11Params(path)
    
    
if __name__ == "__main__":
    main()    
