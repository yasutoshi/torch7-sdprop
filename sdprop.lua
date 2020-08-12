--[[
#
# ==================================================================
# ==                                                              ==
# ==      PLEASE READ 'license.txt' BEFORE USING THIS SOURCE      ==
# ==                                                              ==
# ==================================================================
#
# Copyright (C) 2020 Nippon Telegraph and Telephone Corporation
# Author: Yasutoshi Ida <yasutoshi.ida@ieee.org>
#

A Torch7 implementation of SDProp [1].

[1] Y. Ida et al., "Adaptive Learning Rate via Covariance Matrix Based Preconditioning for Deep Neural Networks", International Joint Conference on Artificial Intelligence(IJCAI), 2017.
URL: https://www.ijcai.org/proceedings/2017/0267.pdf

ARGS:
- 'opfunc'                      : a function that takes a single input (X), the point of a evaluation, and returns f(X) and df/dX
- 'x'                               : the initial point
- 'config`                       : a table with configuration parameters for the optimizer
- 'config.learningRate' : initial learning rate
- 'config.gamma'          : bias for online updating of diagonal covariance matrix
- 'config.epsilon'          : for numerical stability
- 'state'                        : a table describing the state of the optimizer; after each call the state is modified

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update
]]

function optim.sdprop(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.001

    local gamma = config.gamma or 0.99
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)

    -- Initialization
    state.t = state.t or 0
    -- Exponential moving average of first order gradients
    state.mu = state.mu or x.new(dfdx:size()):zero()
    -- Exponential moving average of diagonal covariance matrix
    state.c = state.c or x.new(dfdx:size()):zero()
    -- A tmp tensor to hold the sqrt(c) + epsilon
    state.denom = state.denom or x.new(dfdx:size()):zero()

    state.t = state.t + 1
    
    -- Online update of state.mu and state.c
    local c_dfdx = torch.add(dfdx, -state.mu)
    state.c:mul(gamma):addcmul(gamma*(1-gamma), c_dfdx, c_dfdx)
    state.mu:mul(gamma):add(1-gamma, dfdx)

    state.denom:copy(state.c):sqrt():add(epsilon)

    local biasCorrection = 1 - gamma^state.t
    local stepSize = lr * math.sqrt(biasCorrection)
    -- (2) update x
    x:addcdiv(-stepSize, dfdx, state.denom)

    -- return x*, f(x) before optimization
    return x, {fx}
end
