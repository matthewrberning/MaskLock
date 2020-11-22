#include <iostream>

#include <chrono>
#include <thread>

#include "usb_relay_device - luisversion.h"
#pragma comment(lib, "usb_relay_device.lib")


/// Opens relay for X number of seconds

//

int main(int secondsinput, int lag)

{


	usb_relay_init();
	/// <summary>
	//Is producing a structured list of relaydices
	/// </summary>
	/// <returns></returns>
	/*switch(secondsinput)   // test condition incoming from python
	{
		case 0:
			// case means blanket open 
		case 1: // case means blanket open with  lag before clsoe 
		case 2: // case is for blanket Close
		case 3:// all relayts wiill close after lag

	}
	*/

	if (relaycontrols::listofrelaydevices)
		relaycontrols::openallrelays(10);

	//need to pass a handler to the opener ( handler is the int from this function






}


