/*
 * 3D Tracking System: MASTER RECEIVER ANCHOR 0 (A0)
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Arduino.h>

#define SERIAL_LOG Serial
#define SERIAL_AT mySerial2

HardwareSerial SERIAL_AT(2);

#define UWB_INDEX 0
#define UWB_TAG_COUNT 64

#define RESET 16
#define IO_RXD2 18
#define IO_TXD2 17
#define I2C_SDA 39
#define I2C_SCL 38

Adafruit_SSD1306 display(128, 64, &Wire, -1);

String response = "";
String rec_head = "AT+RANGE";

String sendData(String command, const int timeout, boolean debug);
void range_analy(String data);

void setup()
{
    pinMode(RESET, OUTPUT);
    digitalWrite(RESET, HIGH);

    SERIAL_LOG.begin(115200);
    SERIAL_AT.begin(115200, SERIAL_8N1, IO_RXD2, IO_TXD2);

    SERIAL_AT.println("AT");
    Wire.begin(I2C_SDA, I2C_SCL);
    delay(1000);
    
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        SERIAL_LOG.println(F("SSD1306 allocation failed"));
    }
    display.clearDisplay();

    sendData("AT?", 2000, 1);
    sendData("AT+RESTORE", 5000, 1); 
    sendData("AT+SETCFG=" + String(UWB_INDEX) + ",1,1,1", 2000, 1); 
    sendData("AT+SETCAP=" + String(UWB_TAG_COUNT) + ",10,1", 2000, 1); 
    
    // --- CALIBRATION INTERFACE ---
    sendData("AT+SETANT=16399", 2000, 1); 
    
    sendData("AT+SETRPT=1", 2000, 1);    
    sendData("AT+SAVE", 2000, 1);        
    sendData("AT+RESTART", 2000, 1);     
}

void loop()
{
    while (SERIAL_LOG.available() > 0) {
        SERIAL_AT.write(SERIAL_LOG.read());
        yield();
    }
    while (SERIAL_AT.available() > 0) {
        char c = SERIAL_AT.read();
        if (c == '\r') continue;
        else if (c == '\n') {
            if (response.indexOf(rec_head) != -1) {
                range_analy(response);
            } else {
                SERIAL_LOG.println(response);
            }
            response = "";
        }
        else response += c;
    }
}

void range_analy(String data)
{
    data.replace(" ", ""); 
    int tidIdx = data.indexOf("tid:");
    int rangeIdx = data.indexOf("range:(");
    int ancidIdx = data.indexOf("),ancid:");

    if (tidIdx == -1 || rangeIdx == -1 || ancidIdx == -1) return;

    String id_str = data.substring(tidIdx + 4, data.indexOf(",mask:"));
    String range_str = data.substring(rangeIdx, ancidIdx + 1);

    int range_list[8] = {0};
    int count = sscanf(range_str.c_str(), "range:(%d,%d,%d,%d,%d,%d,%d,%d)",
                       &range_list[0], &range_list[1], &range_list[2], &range_list[3],
                       &range_list[4], &range_list[5], &range_list[6], &range_list[7]);

    if (count == 8) {
        String json_str = "{\"id\":" + id_str + ",\"range\":[";
        for (int i = 0; i < 8; i++) {
            json_str += String(range_list[i]);
            if (i != 7) json_str += ",";
        }
        json_str += "]}";
        SERIAL_LOG.println(json_str);
    }
}

String sendData(String command, const int timeout, boolean debug) {
    String resp = "";
    SERIAL_LOG.println(command);
    SERIAL_AT.println(command);
    long int time = millis();
    while ((time + timeout) > millis()) {
        while (SERIAL_AT.available()) { resp += (char)SERIAL_AT.read(); }
    }
    if (debug) SERIAL_LOG.println(resp);
    return resp;
}