Model guide
===========

jmstate links a longitudinal biomarker process to a multi-state event history
through shared individual random effects.

Longitudinal sub-model
----------------------

Individual observations follow

.. math::

   y_{ij} = h(t_{ij}, \psi_i) + \epsilon_{ij}, \qquad \epsilon_{ij} \sim \mathcal{N}(0, R),

where :math:`h` is a user-defined regression function and

.. math::

   \psi_i = f(\gamma, X_i, b_i), \qquad b_i \sim \mathcal{N}(0, Q).

Multi-state sub-model
---------------------

For a transition :math:`k \to k'` at time :math:`t` after entering the current
state at :math:`t_0`, the hazard is

.. math::

   \lambda^{k \to k'}(t_0, t) = \lambda_0^{k \to k'}(t_0, t) \exp\left(\alpha^{k \to k'} g^{k \to k'}(t, \psi_i) + \beta^{k \to k'} X_i\right).

The state graph can include recurrent, absorbing, and monotone transitions
under a semi-Markov assumption.

Estimation
----------

Parameters are estimated by maximizing the observed-data log-likelihood. The
gradient is evaluated with the Fisher identity and approximated using a
Metropolis-within-Gibbs sampler over the random effects combined with
stochastic gradient optimization.
