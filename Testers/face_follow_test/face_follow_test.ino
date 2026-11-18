#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

const int CH_PAN = 1;    // camera left/right
const int CH_TILT = 2;   // camera up/down
const int SERVO_MIN_US = 500;
const int SERVO_MAX_US = 2500;
const int TILT_MIN = 30;
const int TILT_MAX = 150;

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

int panAngle = 90;
int tiltAngle = 90;
String buf = "";
unsigned long lastCmd = 0;
unsigned long cmdCount = 0;

void writeServo(int channel, int angle) {
  angle = constrain(angle, 0, 180);
  pwm.writeMicroseconds(channel, map(angle, 0, 180, SERVO_MIN_US, SERVO_MAX_US));
}

void handle(String cmd) {
  cmd.trim();
  if (!cmd.startsWith("CAM:")) return;
  int comma = cmd.indexOf(',');
  if (comma <= 4) return;
  panAngle = constrain(cmd.substring(4, comma).toInt(), 0, 180);
  tiltAngle = constrain(cmd.substring(comma + 1).toInt(), TILT_MIN, TILT_MAX);
  writeServo(CH_PAN, panAngle);
  writeServo(CH_TILT, tiltAngle);
  cmdCount++;
  lastCmd = millis();
}

void setup() {
  Serial.begin(115200);
  pinMode(2, OUTPUT);
  Wire.begin();   // SDA = GPIO 21, SCL = GPIO 22
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);
  delay(10);

  // wiring check: small wiggle on both servos at startup
  writeServo(CH_PAN, 90);  writeServo(CH_TILT, 90);  delay(500);
  writeServo(CH_PAN, 70);  writeServo(CH_TILT, 75);  delay(400);
  writeServo(CH_PAN, 110); writeServo(CH_TILT, 105); delay(400);
  writeServo(CH_PAN, 90);  writeServo(CH_TILT, 90);

  Serial.println("Face follow test ready");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handle(buf);
      buf = "";
    } else if (c != '\r' && buf.length() < 32) {
      buf += c;
    }
  }

  // onboard LED lights while commands are arriving (GPIO 2 on most ESP32 dev boards)
  digitalWrite(2, millis() - lastCmd < 100 ? HIGH : LOW);
}
