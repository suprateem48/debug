// Added for "twiddling" the PID coefficients
bool twiddle_on_ = false;
double twiddle_best_error_ = 1000000;
bool twiddle_state_ = 0;
int twiddle_idx = 0;
int twiddle_iterations_ = 0;
std::vector<double> p = {0.27, 0.001, 3.0};
std::vector<double> dp = {0.05, 0.001, 0.05};
void twiddle(PID &pid_control) 
{
  std::cout << "State: " << twiddle_state_ << std::endl;
  std::cout << "PID Error: " << pid_control.TotalError() << ", Best Error: " << twiddle_best_error_ << std::endl;
  
  if (twiddle_state_ == 0) 
  {
    twiddle_best_error_ = pid_control.TotalError();
    p[twiddle_idx] += dp[twiddle_idx];
    twiddle_state_ = 1;
  } 
  else if (twiddle_state_ == 1) 
  {
    if (pid_control.TotalError() < twiddle_best_error_) 
    {
      twiddle_best_error_ = pid_control.TotalError();
      dp[twiddle_idx] *= 1.1;
      twiddle_idx = (twiddle_idx + 1) % 3; //rotate over the 3 vector indices
      p[twiddle_idx] += dp[twiddle_idx];
      twiddle_state_ = 1;
    } 
    else 
    {
      p[twiddle_idx] -= 2 * dp[twiddle_idx];
      if (p[twiddle_idx] < 0) 
      {
        p[twiddle_idx] = 0;
        twiddle_idx = (twiddle_idx + 1) % 3;
      }
      twiddle_state_ = 2;
    }
  } 
  else 
  { //twiddle_state_ = 2
    if (pid_control.TotalError() < twiddle_best_error_) 
    {
      twiddle_best_error_ = pid_control.TotalError();
      dp[twiddle_idx] *= 1.1;
      twiddle_idx = (twiddle_idx + 1) % 3;
      p[twiddle_idx] += dp[twiddle_idx];
      twiddle_state_ = 1;
    } 
    else 
    {
      p[twiddle_idx] += dp[twiddle_idx];
      dp[twiddle_idx] *= 0.9;
      twiddle_idx = (twiddle_idx + 1) % 3;
      p[twiddle_idx] += dp[twiddle_idx];
      twiddle_state_ = 1;
      //pid.Init(p[0], p[1], p[2]);
    }
  }
  pid_control.set_coeffs(p[0], p[1], p[2]);
}
// End Twiddle
