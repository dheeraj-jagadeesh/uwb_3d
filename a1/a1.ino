/*
 * 3D Tracking System: SATELLITE REFERENCE ANCHORS (A1 / A2 / A3)
 */

#include <Arduino.h>
#define SERIAL_LOG Serial
#define SERIAL_AT mySerial2

HardwareSerial SERIAL_AT(2);

// CRITICAL: Change this index to 1, 2, or 3 respectively for your satellite units
#define UWB_INDEX 1 

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
    sendCmd("AT+RESTORE"); // Clear out older conflicting configs [cite: 287]
    delay(2000);
    
    sendCmd("AT+SETCFG=" + String(UWB_INDEX) + ",1,1,1"); // Configure target role Index 
    sendCmd("AT+SETCAP=64,10,1"); // Enable extended mode array transfers [cite: 317, 336]
    sendCmd("AT+SETANT=16399");   // Synchronized structural delay value [cite: 313]
    sendCmd("AT+SETRPT=0");       // Shut down local satellite loop processing outputs [cite: 341]
    sendCmd("AT+SAVE");           // Commit configuration parameters [cite: 297]
    sendCmd("AT+RESTART");        // Cycle node [cite: 279]
}

void loop() {}