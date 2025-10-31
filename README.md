# FAABPsTransportFieldSim

Simulation of a Force-Aligning Active Brownian Particle (FAABP) swarm cooperatively transporting a payload to a goal.

Hyperparameters are defined in the global variables section in main.py, where you can also run the whole simulation from. 

Use run_tests.py to run unit tests for every function + an integration test. This is useful for verifying that everything is working correctly while changing the simulation. It is still good, though, to generate the mp4 of the simulation by running it normally, as a visual confirmation that everything is behaving as expected!

TODO:
- I think there is a good amount of computational redundancy that can be carved away, to make it all run faster.