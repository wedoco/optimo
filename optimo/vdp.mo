model vdp
  "Van der Pol oscillator model with a default controller"
  
  Real x1(start=0, nominal=1) "The first state";  
  Real x2(start=0, nominal=1) "The second state"; 
  input Real u(start=0, nominal=1) "The control signal"; 
  output Real objectiveIntegrand(nominal=1) "The objective integrand signal"; 

equation
  der(x1) = (1 - x2^2) * x1 - x2 + u; 
  der(x2) = x1; 
  objectiveIntegrand = x1^2 + x2^2 + u^2;
  
annotation(
    experiment(StartTime = 0, StopTime = 20, Tolerance = 1e-06, Interval = 0.04));
end vdp;