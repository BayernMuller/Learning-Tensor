# Regularization

Regularization is used to prevent overfitting of the model.


<!-- $$
L(W) = \frac{1}{N}\sum_{i=1}^{N}L_i(f(x_i,W),y_i)+\lambda R(W)
$$ --> 
<br/>
<br/>
<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=L(W)%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7DL_i(f(x_i%2CW)%2Cy_i)%2B%5Clambda%20R(W)" width="80%"></div>
<br/>
<br/>



λR(x) is for regularizaiton, and λ is [**hyper parameter**](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))

### L2, L1 Regularization

<!-- $$
L2: R(W) = \sum _k \sum _l W^2_{k,l}
$$ --> 
<br/>
<br/>
<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=L2%3A%20R(W)%20%3D%20%5Csum%20_k%20%5Csum%20_l%20W%5E2_%7Bk%2Cl%7D" width="50%"></div>
<!-- $$
L1: R(W) = \sum _k \sum _l |W_{k,l}|
$$ --> 
<br/>
<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=L1%3A%20R(W)%20%3D%20%5Csum%20_k%20%5Csum%20_l%20%7CW_%7Bk%2Cl%7D%7C" width="50%"></div>
<br/><br/>


### Drop Out

dropout removes some part of connections of neuron.

![](res/dropout.png)


