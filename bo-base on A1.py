#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Larry, i-MEET, FZJ, 02.02.2025
# A notebook to perform a discontinuos Bayesian Optimization
# Multi-objective version

#import sys,os
#sys.path.append('/home/larry/Documents/vscodeprojs/imeet-ht-features')

#%matplotlib notebook 
#%matplotlib inline

from imeet import *
from imeet.agents.dtmat2 import DTmat2 # new optimization class using BoTorch directly
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood


# In[2]:


# User settings
toy = False # set to false when using an external workflow; True when running the toy model
fn = 'bo-1st base on A1.xlsx' # path to xls sheet containing X and Y data so far
ycols = ['A1','Eg'] # Two objectives: photoinduced phase separation and optical bandgap
n_init = 40 # only for the toy model
minimize = False

# Define dimensions and ranges

if toy:
    # Define dimensions and ranges
    X1 = Fitparam(name='X1', val = 0.5, lims = [0.1,1])
    X2 = Fitparam(name='X2', val = 0.5, lims = [0.1,1])
    params = [X1,X2]
    xcols = [s.name for s in params]

else:
    'Cs(a)','Rb(b)','DMA(c)','Br(n)','Cl(m)'
    X1 = Fitparam(name='Cs(a)', val = 0.2, lims = [0.001,0.5])
    X2 = Fitparam(name='Rb(b)', val = 0.1, lims = [0.001,0.2])
    X3 = Fitparam(name='DMA(c)', val = 0.1, lims = [0.001,0.25])
    X4 = Fitparam(name='Br(n)', val = 0.2, lims = [0.001,0.5])
    X5 = Fitparam(name='Cl(m)', val = 0.1, lims = [0.001,0.2])
    params = [X1,X2,X3,X4,X5]
    xcols = [s.name for s in params]


# In[3]:


# **************** MULTI-OBJECIVE MODEL FUNCTION ***********
# We want to minimize photoinduced phase segregation (PPS) for a given bandgap 
# This toy model assumes:
#
# First objective: PPS. 
#   Quadratic function of Bromine content 
#   second process condition shifts the minimum up, but makes the parabola flatter
#   therefore, you want the second process weak for low bromine but high for large bromine
#
# Second objective: Optical bandgap
#   Increases linearly with Bromine
#   second process condition has no influence

# Define the model hyperparameters
a = Fitparam(name='a', val = 1, lims = [0.2,2])
b = Fitparam(name='b', val = 10, lims = [0.2,2])
hyperparams = [a,b]
noise = [0.005,0.025]
#noise = [0,0] # for testing 
 
def model(X, params, objective='pps'):

    """Model function to be used by the optimizer

    Args:
        x (ndarray): first row: 
        params (list): function coefficients as provided by the optimizer
    Returns:
        y (ndarray)
    """

    rng = np.random.default_rng()

    a = [pp.val for pp in params if pp.name=='a'][0] # find Fitparam named 'X1': the Bromine content (acts on the PPS)
    b = [pp.val for pp in params if pp.name=='b'][0] # find Fitparam named 'X2': some other process condition
    #c = [pp.val for pp in params if pp.name=='c'][0] # find Fitparam named 'X3': currently unused
    #d = [pp.val for pp in params if pp.name=='d'][0] # find Fitparam named 'X4': currently unused
    b1 = b*X[:,1] # b1 depends only on temperature
    pps = 0.01*((a/b1)*X[:,0]**2 + b1) + rng.normal(0,noise[0],X[:,0].shape) # first objective: PPS
    pps = -pps + 0.1 #lets maximize the first objective and let it become positive
    #y = (a+b1+c)*X[:,0]#+c+d # therefore the linear slope along X[:,0] depends on temperature
    eg = 2.4 - exp(-X[:,0]) + rng.normal(0,noise[1],X[:,0].shape) # second objective: NEGATIVE bandgap so we can minimize all objectives
    eg = eg # lets maximize the second objective
    return pps if objective=='pps' else eg


# In[4]:


# get initial recommendations (SOBOL) 
# SKIP IF ALREADY DONE

if os.path.exists(fn):
    print(f"Error: File '{fn}' exists. Producing initial recommendations would overwrite it.")

else:
    if toy:
        n_init = n_init # Number of initial points. If None, optimizer will use 2 X number of dimensions
        dm = DTmat2(params=params, n_init=n_init, minimize=minimize) # start the Bayesian Optimization library
        df = dm.initial_points(ycols=ycols) 
        df.to_excel(fn,index=False) # save to Excel file
    else:
        fno = '../Data/Recipes88_Fitting_parameters_monoexponential.csv'
        #fno = '/home/larry/Documents/People/Zijian/Recipes80_Fitting_parameters_monoexponential(1).csv'
        df_0=pd.read_csv(fno)
        df=df_0[(df_0['Eg']>1.6)&(df_0['Eg']<2.2)]
        df = df[abs(df['A1'])<1]
        print(df.shape)
        df1 = df[xcols+ycols]
        df1['A1'] = -abs(df1['A1'])# + 0.15 # to make it positive
        df1['Eg'] = df1['Eg']
        df1['hot']=0
        df1.to_excel(fn,index=False)


# In[5]:


if toy:
    X_new = df.loc[df['hot']==1,xcols].to_numpy()
    pps_next = model(X_new,params=hyperparams,objective='pps') # we produce Y_next by a fake function
    egopt_next = model(X_new,params=hyperparams,objective='eg') # we produce Y_next by a fake function

    #new_values = {'PPS': pps_next, 'Egopt':egopt_next, 'hot': [1 for _ in pps_next]}
    df.loc[df['hot']==1,'A1'] = pps_next # add new evaluations to dataset: obj. 1
    df.loc[df['hot']==1,'Eg'] = egopt_next # add new evaluations to dataset: obj. 2
    df.loc[df['hot']==1,'hot'] = 0 # and set to cold
    df.to_excel(fn,index=False)


# In[6]:


# get recommendations for next batch and write to Excel

batch_size = 8 # Number of recommendations
reference_point = [-0.06,1.6]
num_points = 128
kernel_design = True
beta = None # None: use normal acq_func; float: use exploration incentivating wrapper
# BUG! Don't use float! Incentivating wrapper by ChatGPT o3-mini-high crashes !!!!!! 

dm = DTmat2(params=params, n_init=n_init, batch_size=batch_size, 
            minimize=minimize,fn_sqlite='db.sqlite') # start the Bayesian Optimization library

if kernel_design:
    dm.likelihood = GaussianLikelihood(noise_constraint=Interval(1e-3, 10000)) # WAS (1e-8,1e-3)
    #dm.covar_module = ScaleKernel(
    #        RBFKernel(ard_num_dims=dm.dim, lengthscale_constraint=Interval(0.005, 4.0))+
    #        LinearKernel(num_dimensions=dm.dim),
    #    outputscale_constraint=Interval(0.1,10)
    #)

    dm.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=dm.dim, lengthscale_constraint=Interval(0.0005, 40.0)),
        outputscale_constraint=Interval(0.01,100)
    )


df = dm.dataset_from_excel(fn,ycols=ycols) 
# if your objectives are noisy (experiment), choose 'qNEHVI', else (Bayesian Inference) choose 'qEHVI' 
X_next = dm.new_batch_mo(un_normalize=True,
                         ref_point=reference_point, # in NATURAL DOMAIN
                         acq_func_name='qNEHVI',
                         num_points=num_points,
                         beta=beta) 
xcols = [pp.name for pp in dm.params if pp.relRange!=0] 
new_values = {xc:X_next[:,ii] for ii,xc in enumerate(xcols)}
hot = [1 for _ in range(X_next.shape[0])]
new_rows = {k:v.numpy() for k,v in new_values.items()}
new_rows |= {yc:0 for yc in ycols}
new_rows |= {'hot':hot}

df1 = pd.DataFrame(new_rows)
df = pd.concat([df, df1], ignore_index=True)
df.to_excel(fn,index=False)


# In[7]:


# diagnostics
#r = dm.probe_model() # learned output scale, length scales, and noise
#for rr in r:
#    print(rr)
axx = dm.plot_partial_dependence(objectives = ycols,X_next=X_next)
dm.plot_pareto_frontier(iobj=[0,1],objectives = ycols,X_next=X_next,show_uncertainties=True)
dm.plot_result_plot()


# In[ ]:


for m in dm.model.models:
    print('Variance',m.covar_module.base_kernel.kernels[1].variance.item())
    print('Length scale',m.covar_module.base_kernel.kernels[0].lengthscale)
    print('Noise',m.likelihood.noise.item())


# In[ ]:





# In[ ]:




