/*
 * Mecanum Robot Motor Control - Arduino Mega
 * 4x BTS7960 Motor Drivers
 * Serial from Pi4 @ 115200 baud
 *
 * Protocol:
 *   FL <speed>\n   → Front-Left motor (-255..255)
 *   FR <speed>\n   → Front-Right motor
 *   BL <speed>\n   → Back-Left (Rear-Left) motor
 *   BR <speed>\n   → Back-Right (Rear-Right) motor
 *   M <fl> <fr> <bl> <br>\n → all four at once
 *   S\n            → emergency stop
 *   P\n            → ping (replies PONG)
 *
 * BTS7960 Pin Mapping:
 *   Motor        LPWM  RPWM  L_EN  R_EN
 *   Front-Left     2     3    22    23
 *   Front-Right    4     5    24    25
 *   Rear-Left      6     7    26    27
 *   Rear-Right     8     9    28    29
 */

struct Motor {
  uint8_t lpwm;
  uint8_t rpwm;
  uint8_t len;
  uint8_t ren;
};

enum { FL = 0, FR = 1, BL = 2, BR = 3, NUM_MOTORS = 4 };

Motor motors[NUM_MOTORS] = {
  { 2, 3, 22, 23 },  // Front-Left
  { 4, 5, 24, 25 },  // Front-Right
  { 6, 7, 26, 27 },  // Rear-Left
  { 8, 9, 28, 29 },  // Rear-Right
};

void setMotor(uint8_t idx, int16_t speed) {
  if (idx >= NUM_MOTORS) return;
  speed = constrain(speed, -255, 255);
  Motor &m = motors[idx];

  // Enable driver
  digitalWrite(m.len, HIGH);
  digitalWrite(m.ren, HIGH);

  if (speed > 0) {
    analogWrite(m.lpwm, speed);
    analogWrite(m.rpwm, 0);
  } else if (speed < 0) {
    analogWrite(m.lpwm, 0);
    analogWrite(m.rpwm, -speed);
  } else {
    analogWrite(m.lpwm, 0);
    analogWrite(m.rpwm, 0);
    digitalWrite(m.len, LOW);
    digitalWrite(m.ren, LOW);
  }
}

void stopAll() {
  for (uint8_t i = 0; i < NUM_MOTORS; i++) setMotor(i, 0);
}

char buf[64];
uint8_t bufIdx = 0;

void processCommand(char *cmd) {
  while (*cmd == ' ') cmd++;

  // Single motor commands
  if (strncmp(cmd, "FL", 2) == 0) {
    setMotor(FL, atoi(cmd + 2));
    Serial.print("OK FL "); Serial.println(atoi(cmd + 2));
  }
  else if (strncmp(cmd, "FR", 2) == 0) {
    setMotor(FR, atoi(cmd + 2));
    Serial.print("OK FR "); Serial.println(atoi(cmd + 2));
  }
  else if (strncmp(cmd, "BL", 2) == 0) {
    setMotor(BL, atoi(cmd + 2));
    Serial.print("OK BL "); Serial.println(atoi(cmd + 2));
  }
  else if (strncmp(cmd, "BR", 2) == 0) {
    setMotor(BR, atoi(cmd + 2));
    Serial.print("OK BR "); Serial.println(atoi(cmd + 2));
  }

  // All four motors
  else if (cmd[0] == 'M' && cmd[1] == ' ') {
    char *p = cmd + 2;
    int16_t speeds[4];
    for (uint8_t i = 0; i < 4; i++) {
      speeds[i] = (int16_t)strtol(p, &p, 10);
    }
    for (uint8_t i = 0; i < 4; i++) setMotor(i, speeds[i]);
    Serial.print("OK M ");
    for (uint8_t i = 0; i < 4; i++) { Serial.print(speeds[i]); Serial.print(" "); }
    Serial.println();
  }

  // Stop
  else if (cmd[0] == 'S' || cmd[0] == 's') {
    stopAll();
    Serial.println("OK STOP");
  }

  // Ping
  else if (cmd[0] == 'P' || cmd[0] == 'p') {
    Serial.println("PONG");
  }

  else {
    Serial.print("ERR: "); Serial.println(cmd);
  }
}

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    pinMode(motors[i].lpwm, OUTPUT);
    pinMode(motors[i].rpwm, OUTPUT);
    pinMode(motors[i].len, OUTPUT);
    pinMode(motors[i].ren, OUTPUT);
  }
  stopAll();

  Serial.println("READY - Commands: FL/FR/BL/BR <speed>, M <fl fr bl br>, S=stop, P=ping");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (bufIdx > 0) {
        buf[bufIdx] = '\0';
        processCommand(buf);
        bufIdx = 0;
      }
    } else if (bufIdx < sizeof(buf) - 1) {
      buf[bufIdx++] = c;
    }
  }
}