#include "PID.h"

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() 
{
  Kp = 0;
  Ki = 0;
  Kd = 0;
}

PID::PID(double Kp_, double Ki_, double Kd_) 
{
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;
}

PID::~PID() {}

/*
void PID::Init(double Kp_, double Ki_, double Kd_) {
  
  // TODO: Initialize PID coefficients (and errors, if needed)
   
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;
}
*/

void PID::UpdateError(double cte, double prev_cte = 0, double total_cte = 0) 
{
  /**
   * TODO: Update PID errors based on cte.
   */
  p_error = Kp * cte;
  d_error = Kd * (cte - prev_cte);
  i_error = Ki * total_cte;
}

double PID::TotalError() 
{
  /**
   * TODO: Calculate and return the total error
   */

  return -1.0 * (p_error + d_error + i_error); // TODO: Add your total error calc here!
}