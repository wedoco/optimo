model vdp
  "Van der Pol oscillator model with a default controller"
  
  Real x1(start=0, fixed=true) "The first state";  
  Real x2(start=1, fixed=true) "The second state"; 
  input Real u "The control signal"; 

equation
  der(x1) = (1 - x2^2) * x1 - x2 + u; 
  der(x2) = x1; 
  
annotation(
    experiment(StartTime = 0, StopTime = 20, Tolerance = 1e-06, Interval = 0.04));
end vdp;