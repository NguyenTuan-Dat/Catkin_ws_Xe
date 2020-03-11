#include <string.h>
#include <termios.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

#include "JHPWMPCA9685.h"

//using namespace std;

/* ESC facts calculations:
    At 61 hz a frame is 16.39 milliseconds
    
    According to the oscilloscope measure the pulse 
    from the Traxxas Desert's RF reciver/controller

    Full Reverse signal to the ESC is 1 ms
    Neutral signal to the ESC is 1.5 ms
    Full Throttle signal to the ESC is 2 mss
*/

// PCA9685 pwm frequency base on ESC
#define PWM_FREQ 100

/* Duty cycle
    PCA9685 has 12 bits resolution: 4096
    Neutral Duty 15% 
    Max_Forward 20%
    Max_Reverse 10%
    
    Servo EMAX at 50Hz
    0           3.6%
    180        12.8%
*/
#define THROTTLE_MAX_REVERSE   410
#define THROTTLE_NEUTRAL       640  
#define THROTTLE_MAX_FORWARD   819

#define STEERING_MAX_RIGHT     460 
#define STEERING_MAX_LEFT      860

#define CAMERA_MAX_RIGHT      299 // 147  
#define CAMERA_MAX_LEFT       1050 // 524


// The THROTTLE is plugged into the following PWM channel
#define THROTTLE_CHANNEL  1
// The Steering Servo is plugged into the following PWM channel
#define STEERING_CHANNEL 0
// The Camera Holder servo is plugged into the following PWM channel
#define CAMERA_CHANNEL 2

#define MIN_ANGLE -60 // 10 * maxLeft angle(20 degree) = -200, mul 10 to smooth control
#define MAX_ANGLE 60  // 10 * maxRight angle

#define MIN_SERVO 0
#define MAX_SERVO 180

#define MOTOR_A 5
#define MOTOR_B 4
#define MOTOR_C 3
#define MOTOR_D 2

#define EN_AB 6
#define EN_CD 7
int map (double x, int in_min, int in_max, int out_min, int out_max);
void api_pwm_pca9685_init( PCA9685 *pca9685);
int api_set_FORWARD_control(PCA9685 *pca9685,double &throttle_val);
int api_set_BRAKE_control( PCA9685 *pca9685,double &throttle_val);
int api_set_STEERING_control(PCA9685 *pca9685,double &theta);
int api_set_CAMERA_control( PCA9685 *pca9685,double &theta);
int api_set_left_motor(PCA9685 *pca9685, double &cmd);
int api_set_right_motor(PCA9685 *pca9685, double &cmd);
