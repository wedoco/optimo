model vdp
  "Van der Pol oscillator model with a default controller"
  
  Real x1(start=0) "The first state";  
  Real x2(start=1) "The second state"; 
  input Real u(start=0) "The control signal"; 
  output Real objectiveIntegrand(start=0) "The objective integrand signal"; 

equation
  der(x1) = (1 - x2^2) * x1 - x2 + u; 
  der(x2) = x1; 
  objectiveIntegrand = x1^2 + x2^2 + u^2;

end vdp;