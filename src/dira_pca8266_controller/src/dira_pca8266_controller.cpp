
#include "DiRa_Traxxas_Controller.h"
#include "ros/ros.h"
#include "math.h"

int direction;
int map (double x, int in_min, int in_max, int out_min, int out_max)
{
    double toReturn =  1.0 * (x - in_min) * (out_max - out_min) /
            (in_max - in_min) + out_min ;
    return (int)(round(toReturn));
}
void api_pwm_pca9685_reinit(PCA9685 *pca9685)
{
	pca9685->setAllPWM(0,0) ;
    pca9685->reset() ;
    pca9685->setPWMFrequency(PWM_FREQ);
	sleep(1);
}
void api_pwm_pca9685_init( PCA9685 *pca9685)
{
    // Initialize the PWM board
    int err = pca9685->openPCA9685();
    if (err < 0)
    {
        std::cout<< std::endl<< "Error: %d"<< pca9685->error<< std::flush;
    }

    std::cout<< std::endl<< "PCA9685 Device Address: 0x"<< std::hex
        << pca9685->kI2CAddress<< std::dec<< std::endl;

    pca9685->setAllPWM(0,0) ;
    pca9685->reset() ;
    pca9685->setPWMFrequency(PWM_FREQ) ;
    // Set the PWM to "neutral" (1.5ms)
    sleep(1) ;
    direction = 0;
    int pwm_steer_middle = map( 0, MIN_ANGLE, MAX_ANGLE, STEERING_MAX_RIGHT, STEERING_MAX_LEFT );
    int pwm_camera_middle = map( 0, MIN_SERVO, MAX_SERVO, CAMERA_MAX_RIGHT, CAMERA_MAX_LEFT );
    ROS_INFO("dir value:%d",direction);
    //pca9685->setPWM(CAMERA_CHANNEL, 0, pwm_camera_middle);
    pca9685->setPWM(STEERING_CHANNEL, 0, pwm_steer_middle);
    pca9685->setPWM(THROTTLE_CHANNEL, 0, THROTTLE_NEUTRAL);
    sleep(1) ;
}

int api_set_FORWARD_control( PCA9685 *pca9685,double &throttle_val)
{       
	int err = pca9685->openPCA9685();
	
	//printf("ERROR: %d",err);

    if (err < 0)
    {
        api_pwm_pca9685_reinit(pca9685);
    }


    if(throttle_val > 0)
    {
	if(direction == 1)
	{
		// direction = 1;
        int pwm = map( throttle_val, 0, 100, THROTTLE_NEUTRAL, THROTTLE_MAX_FORWARD );
	//if(pwm > THROTTLE_MAX_FORWARD)
	//{
	//	pwm = THROTTLE_MAX_FORWARD;
	//}
		ROS_INFO("pwm value:%d",pwm);
        pca9685->setPWM(THROTTLE_CHANNEL,0, pwm);
	ROS_INFO("INSIDE forward dir value:%d pwm value:%d",direction, pwm);
				
	}
	//else if(direction == -1 && throttle_val <=20)
	//{
	//	pca9685->setPWM(THROTTLE_CHANNEL,0, THROTTLE_MAX_FORWARD);
	//	ROS_INFO("dir value:%d",direction);
        //	usleep(187500);
	//	pca9685->setPWM(THROTTLE_CHANNEL,0, THROTTLE_NEUTRAL);
	//	direction = 0;
	//	ROS_INFO("dir value:%d",direction);
	//	usleep(187500);
	//	int pwm = map(15, 0, 100, THROTTLE_NEUTRAL, THROTTLE_MAX_FORWARD );
	//	pca9685->setPWM(THROTTLE_CHANNEL,0, pwm);
	//	usleep(187500);
	//	
	//}
        direction = 1;
    //     int pwm = map( throttle_val, 0, 100, THROTTLE_NEUTRAL, THROTTLE_MAX_FORWARD );
	// //if(pwm > THROTTLE_MAX_FORWARD)
	// //{
	// //	pwm = THROTTLE_MAX_FORWARD;
	// //}
	
    //     pca9685->setPWM(THROTTLE_CHANNEL,0, pwm);
	// ROS_INFO("forward dir value:%d pwm value:%d",direction, pwm);
    }
    else if(throttle_val < 0)
    {
		ROS_INFO("SUPPPPPPPP BITCH?");	
		if(direction == -1)
		{
			// pca9685->setPWM(THROTTLE_CHANNEL,0, THROTTLE_MAX_REVERSE);
			// ROS_INFO("dir value:%d",direction);
			// usleep(187500);
			// pca9685->setPWM(THROTTLE_CHANNEL,0, THROTTLE_NEUTRAL);
			// direction = 0;
			// ROS_INFO("dir value:%d",direction);
			// usleep(187500);

			// direction = -1;
			int pwm = 4095 - map( abs(throttle_val), 0, 100 , 4095 - THROTTLE_NEUTRAL , 4095 - THROTTLE_MAX_REVERSE);
		//if(pwm > (4095 - THROTTLE_MAX_REVERSE))
		//{
		//	pwm = THROTTLE_MAX_REVERSE;
		//}
			pca9685->setPWM(THROTTLE_CHANNEL,0, pwm);
		ROS_INFO("revese dir value:%d pwm value:%d",direction, pwm);
		}
		direction = -1;
	}
	
	//else if(direction == 1 && throttle_val >= -20)
	//{
	//	pca9685->setPWM(THROTTLE_CHANNEL,0, THROTTLE_MAX_REVERSE);
	//	ROS_INFO("dir value:%d",direction);
	//	usleep(187500);
	//	pca9685->setPWM(THROTTLE_CHANNEL,0, THROTTLE_NEUTRAL);
	//	direction = 0;
	//	ROS_INFO("dir value:%d",direction);
	//	usleep(187500);
	//	int pwm = 4095 - map( abs(-20), 0, 100 , 4095 - THROTTLE_NEUTRAL , 4095 - THROTTLE_MAX_REVERSE);
	//	pca9685->setPWM(THROTTLE_CHANNEL,0, pwm);
	//	usleep(187500);
	//}
        
	pca9685->closePCA9685();
}

int api_set_BRAKE_control( PCA9685 *pca9685,double &throttle_val)
{
	int err = pca9685->openPCA9685();
	//printf("ERROR: %d",err);

    if (err < 0)
    {
        api_pwm_pca9685_reinit(pca9685);
    }

    if(direction == 0)
    {
        pca9685->setPWM(THROTTLE_CHANNEL,0, THROTTLE_NEUTRAL);
	ROS_INFO("dir value:%d",direction);
        usleep(2000);
	direction == 0;
    }
    else if(direction == 1)
    {
        pca9685->setPWM(THROTTLE_CHANNEL,0, 4095 - map(20, 0, 100 , 4095 - THROTTLE_NEUTRAL , 4095 - THROTTLE_MAX_REVERSE));
	ROS_INFO("dir value:%d",direction);
        usleep(100000);
	pca9685->setPWM(THROTTLE_CHANNEL,0, THROTTLE_NEUTRAL);
	ROS_INFO("dir value:%d",direction);
	direction = 0;
	usleep(100000);
    }
    else if(direction == -1)
    {
        pca9685->setPWM(THROTTLE_CHANNEL,0, map( throttle_val, 0, 100, THROTTLE_NEUTRAL, THROTTLE_MAX_FORWARD ));
	ROS_INFO("dir value:%d",direction);
        usleep(100000);
	pca9685->setPWM(THROTTLE_CHANNEL,0, THROTTLE_NEUTRAL);
	ROS_INFO("dir value:%d",direction);
	direction = 0;
	usleep(100000);
    }
	pca9685->closePCA9685();
}

int api_set_STEERING_control( PCA9685 *pca9685,double &theta)
{
	int err = pca9685->openPCA9685();
    //printf("ERROR: %d",err);
    if (err < 0)
    {
        api_pwm_pca9685_reinit(pca9685);
    }

    if( theta < MIN_ANGLE)
        theta = MIN_ANGLE;

    if( theta > MAX_ANGLE )
        theta = MAX_ANGLE;
        
	int pwm1 = map( theta, MIN_ANGLE, MAX_ANGLE, STEERING_MAX_RIGHT, STEERING_MAX_LEFT); 
    pca9685->setPWM(STEERING_CHANNEL,0, pwm1);
	pca9685->closePCA9685();
    return pwm1;
	
}
int api_set_CAMERA_control( PCA9685 *pca9685,double &theta)
{
    if( theta < MIN_SERVO)
        theta = MIN_SERVO;

    if( theta > MAX_SERVO )
        theta = MIN_SERVO;
        
    int pwm2 = map( theta, MIN_SERVO, MAX_SERVO, CAMERA_MAX_RIGHT, CAMERA_MAX_LEFT ); 
    pca9685->setPWM(CAMERA_CHANNEL,0, pwm2);
    return pwm2;
}

int api_set_left_motor(PCA9685 *pca9685, double &cmd)
{	
	int pwm;
	//set A & B
	if(cmd > 0)
	{
		pca9685->setPWM(MOTOR_A, 0, 0);
		pca9685->setPWM(MOTOR_B, 0, 4095);
		pca9685->setPWM(EN_AB, 0, pwm = map(cmd, 0, 100, 0, 4095));
		ROS_INFO("left A0 B1 %d", pwm);
	}
	else if(cmd < 0)
	{
		pca9685->setPWM(MOTOR_B, 0, 0);
		pca9685->setPWM(MOTOR_A, 0, 4095);
		pca9685->setPWM(EN_AB, 0, pwm = map(abs(cmd), 0, 100, 0, 4095));
		ROS_INFO("left A1 B0 %d", pwm);
	}
	else
	{
		pca9685->setPWM(MOTOR_B, 0, 0);
		pca9685->setPWM(MOTOR_A, 0, 0);
		pca9685->setPWM(EN_AB, 0, 0);
		ROS_INFO("left A0 B0 %d", pwm);
	}
	
	return pwm;
}
int api_set_right_motor(PCA9685 *pca9685, double &cmd)
{	
	int pwm;
	//set C & D
	if(cmd > 0)
	{
		pca9685->setPWM(MOTOR_C, 0, 0);
		pca9685->setPWM(MOTOR_D, 0, 4095);
		pca9685->setPWM(EN_CD, 0, pwm = map(cmd, 0, 100, 0, 4095));
		ROS_INFO("right C0 D1 %d", pwm);
	}
	else if(cmd < 0)
	{
		pca9685->setPWM(MOTOR_C, 0, 4095);
		pca9685->setPWM(MOTOR_D, 0, 0);
		pca9685->setPWM(EN_CD, 0, pwm = map(abs(cmd), 0, 100, 0, 4095));
		ROS_INFO("right C1 D0 %d", pwm);
	}
	else
	{
		pca9685->setPWM(MOTOR_C, 0, 0);
		pca9685->setPWM(MOTOR_D, 0, 0);
		pca9685->setPWM(EN_CD, 0, 0);
		ROS_INFO("right C0 D0 %d", pwm);
	}
	
	return pwm;
}
