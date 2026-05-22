/*
 * 3D Tracking System: WEARABLE HAND TARGET TAGS (T0 / T1)
 */

#include <Arduino.h>
#define SERIAL_LOG Serial
#define SERIAL_AT mySerial2

HardwareSerial SERIAL_AT(2);

// CRITICAL: Set to 0 for the Left Hand Module and 1 for the Right Hand Module
#define UWB_INDEX 0 

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
    
    sendCmd("AT+SETCFG=" + String(UWB_INDEX) + ",0,1,1"); // Role: 0 (Mobile Tag Unit) 
    sendCmd("AT+SETCAP=64,10,1"); // Match base-station capacity properties [cite: 317, 336]
    sendCmd("AT+SETANT=16399");   // Sync calibrated antenna parameter value [cite: 313]
    sendCmd("AT+SETRPT=1");       // Enable automated ranges [cite: 341]
    sendCmd("AT+SAVE");           // Flash save parameters [cite: 297]
    sendCmd("AT+RESTART");        // Reboot module [cite: 279]
}

void loop() {}