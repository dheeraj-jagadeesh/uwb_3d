/*
 * Full-Body Setup: HEAD TAG (T4)
 */

#include <Arduino.h>
#define SERIAL_LOG Serial
#define SERIAL_AT mySerial2

HardwareSerial SERIAL_AT(2);

#define UWB_INDEX 4 // Set uniquely for Head Mount

#define RESET 16
#define IO_RXD2 18
#define IO_TXD2 17

void sendCmd(String cmd) {
    SERIAL_LOG.println(cmd);
    SERIAL_AT.println(cmd);
    delay(500);
}

void setup() {
    pinMode(RESET, OUTPUT);
    digitalWrite(RESET, HIGH);
    SERIAL_LOG.begin(115200);
    SERIAL_AT.begin(115200, SERIAL_8N1, IO_RXD2, IO_TXD2);
    delay(2000);

    sendCmd("AT?");
    sendCmd("AT+RESTORE"); 
    delay(2000);
    
    sendCmd("AT+SETCFG=" + String(UWB_INDEX) + ",0,1,1"); // Role: 0 (Tag Unit)
    sendCmd("AT+SETCAP=64,10,1"); 
    sendCmd("AT+SETANT=16399");   // Calibrated parameter delay
    sendCmd("AT+SETRPT=1");       
    sendCmd("AT+SAVE");           
    sendCmd("AT+RESTART");        
}

void loop() {}