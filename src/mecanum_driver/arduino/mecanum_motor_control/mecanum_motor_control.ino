/*
 * Mecanum Robot Motor Control - Arduino Mega 2560
 *
 * Receives wheel speed commands from the Raspberry Pi 5 over USB serial
 * and drives 4 motors via 4x BTS7960 H-bridge drivers.  In the Two-Tier
 * SBC + MCU layout the Mega ALSO reads the BNO055 9-DOF IMU over I2C so
 * that wheel-encoder ticks and IMU orientation/angular-velocity/linear-
 * acceleration share a single MCU-side timestamp.  This removes the Pi 5
 * I2C clock-stretching problems that the BNO055 had when the IMU was
 * hosted directly on the Pi.
 *
 * Serial Protocol:
 *   Command IN:  "M <fl> <fr> <rl> <rr>\n"  (-255 to 255)
 *   Response OUT: "OK <fl> <fr> <rl> <rr>\n"
 *   Stop:         "S\n"  = emergency stop  (-> "STOPPED\n")
 *   Ping:         "P\n"  = liveness probe  (-> "PONG\n")
 *
 *   Unified telemetry (~20 Hz when USE_ENCODERS+USE_BNO055 are enabled):
 *     "ODOM <fl_ticks> <fr_ticks> <rl_ticks> <rr_ticks>"
 *          " <qw> <qx> <qy> <qz>"
 *          " <gx> <gy> <gz>"
 *          " <ax> <ay> <az>\n"
 *
 *   Where:
 *     - tick counts are cumulative signed integers since boot
 *     - the quaternion is the BNO055 fused orientation (unitless)
 *     - the gyro vector is in deg/s  (the Pi-side driver converts to rad/s)
 *     - the accel vector is in m/s^2 (linear acceleration WITH gravity)
 *
 * The legacy "E <fl> <fr> <rl> <rr>\n" line is emitted only when the
 * BNO055 is disabled / not detected, so existing tooling that only cares
 * about wheel ticks keeps working in a no-IMU bench setup.
 *
 * BTS7960 Driver Wiring - ADJUST TO YOUR WIRING:
 * ─────────────────────────────────────────────────────────
 *  Motor          RPWM(fwd)  LPWM(rev)  R_EN  L_EN
 * ─────────────────────────────────────────────────────────
 *  Front-Left      2          3          4     4  (tied)
 *  Front-Right     5          6          7     7  (tied)
 *  Rear-Left       8          9         10    10  (tied)
 *  Rear-Right     11         12         13    13  (tied)
 * ─────────────────────────────────────────────────────────
 *
 * BTS7960 logic:
 *   Forward:  RPWM = speed (0-255), LPWM = 0
 *   Reverse:  RPWM = 0,            LPWM = speed (0-255)
 *   Brake:    RPWM = 0,            LPWM = 0
 *   R_EN and L_EN must be HIGH to enable the driver.
 *   They can be tied together to a single enable pin per driver.
 *
 * BNO055 WIRING (I2C bus shared with the Mega):
 *   SDA -> Mega pin 20
 *   SCL -> Mega pin 21
 *   VIN -> 3.3 V (or 5 V if your breakout includes a regulator — most do)
 *   GND -> GND
 *   ADR -> floating  (default address 0x28) or VCC (0x29)
 *
 *   IMPORTANT: The Mega's I2C runs at 5 V logic.  Most BNO055 breakouts
 *   (Adafruit, generic) include level shifters AND a 3.3 V regulator, so
 *   wiring is direct.  If you have a bare module, add 3.3 V <-> 5 V level
 *   shifters on SDA/SCL or the chip will be flaky.
 *
 * ENCODER SUPPORT:
 *   The Arduino Mega has 6 external interrupt pins (2, 3, 18, 19, 20, 21)
 *   and plenty of digital pins.  Pins 20-21 are reserved for I2C when the
 *   BNO055 is enabled, so we use 18/19 + 2/3 for encoder Channel A:
 *     FL: ChA = 18 (INT5), ChB = 22
 *     FR: ChA = 19 (INT4), ChB = 23
 *     RL: ChA =  2 (INT0), ChB = 24
 *     RR: ChA =  3 (INT1), ChB = 25
 *   Pins 2/3 are also the default FL_RPWM/FL_LPWM, so remap them if you
 *   want encoders on all 4 wheels (see notes below).  The default build
 *   leaves USE_ENCODERS uncommented for FL/FR only, which is the common
 *   configuration on the AI-SHA chassis.
 */

// ── Feature flags ────────────────────────────────────────────────────
// Enable encoder reading on the Mega (see pin notes above).
#define USE_ENCODERS
// Enable the BNO055 IMU on the I2C bus.  When defined, the firmware
// emits the unified "ODOM …" telemetry packet instead of the legacy
// "E …" encoder-only packet.  Comment this out for bench testing
// without an IMU connected.
#define USE_BNO055

#ifdef USE_BNO055
  #include <Wire.h>
  #include <Adafruit_Sensor.h>
  #include <Adafruit_BNO055.h>
  #include <utility/imumaths.h>

  // 0x28 = ADR floating, 0x29 = ADR tied to VCC.  Sensor ID 55 is the
  // value Adafruit's library uses for runtime identification.
  Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
  bool bnoReady = false;
#endif

// ── BTS7960 Pin Definitions ──────────────────────────────────────────
// Front-Left Motor
#define FL_RPWM  2
#define FL_LPWM  3
#define FL_EN    4

// Front-Right Motor
#define FR_RPWM  5
#define FR_LPWM  6
#define FR_EN    7

// Rear-Left Motor
#define RL_RPWM  8
#define RL_LPWM  9
#define RL_EN   10

// Rear-Right Motor
#define RR_RPWM 11
#define RR_LPWM 12
#define RR_EN   13

// ── Encoder Pin Definitions (Mega interrupt-capable pins) ────────────
#ifdef USE_ENCODERS
#define FL_ENC_A  18   // INT5
#define FL_ENC_B  22
#define FR_ENC_A  19   // INT4
#define FR_ENC_B  23
#define RL_ENC_A  20   // INT3 — shared with I2C SDA when BNO055 enabled,
                       //         see remap note below.
#define RL_ENC_B  24
#define RR_ENC_A  21   // INT2 — shared with I2C SCL when BNO055 enabled.
#define RR_ENC_B  25

// IMPORTANT: When USE_BNO055 is defined, pins 20-21 are used by the
// hardware I2C bus and CANNOT also drive encoder interrupts.  If you
// need full 4-wheel encoder feedback alongside the BNO055, rewire RL/RR
// Channel A to pins 2 and 3 (and move FL_RPWM/FL_LPWM off those pins
// to e.g. 44/45 which support PWM on the Mega).  The two FL/FR encoders
// alone are enough to drive the existing mecanum forward-kinematics
// estimator, since rear-wheel slip is dominated by the IMU yaw correction.

volatile long encFL = 0, encFR = 0, encRL = 0, encRR = 0;
#endif  // USE_ENCODERS

#if defined(USE_ENCODERS) || defined(USE_BNO055)
  #define TELEMETRY_REPORT_MS  50   // 20 Hz unified ODOM / E packet
  unsigned long lastTelemetryReport = 0;
#endif

#define CMD_TIMEOUT_MS  1000
#define SERIAL_BAUD     115200

unsigned long lastCmdTime = 0;

// Static serial buffer — avoids Arduino String heap fragmentation.
// The ATmega2560 has only 8KB SRAM; String's dynamic reallocation on
// every += character fragments the heap over hours of continuous
// operation, eventually freezing the MCU with motors locked in their
// last state.  A fixed char array uses zero heap.
#define CMD_BUF_SIZE 65  // 64 chars + null terminator
char cmdBuf[CMD_BUF_SIZE];
byte cmdIdx = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);

  // Motor driver pins
  int motorPins[] = {FL_RPWM, FL_LPWM, FL_EN,
                     FR_RPWM, FR_LPWM, FR_EN,
                     RL_RPWM, RL_LPWM, RL_EN,
                     RR_RPWM, RR_LPWM, RR_EN};

  for (int i = 0; i < 12; i++) {
    pinMode(motorPins[i], OUTPUT);
  }

  // Enable all BTS7960 drivers (R_EN and L_EN tied together)
  digitalWrite(FL_EN, HIGH);
  digitalWrite(FR_EN, HIGH);
  digitalWrite(RL_EN, HIGH);
  digitalWrite(RR_EN, HIGH);

#ifdef USE_ENCODERS
  // Encoder channel A pins (interrupt-driven)
  pinMode(FL_ENC_A, INPUT_PULLUP);
  pinMode(FL_ENC_B, INPUT_PULLUP);
  pinMode(FR_ENC_A, INPUT_PULLUP);
  pinMode(FR_ENC_B, INPUT_PULLUP);
  pinMode(RL_ENC_A, INPUT_PULLUP);
  pinMode(RL_ENC_B, INPUT_PULLUP);
  pinMode(RR_ENC_A, INPUT_PULLUP);
  pinMode(RR_ENC_B, INPUT_PULLUP);

  // CHANGE on Channel A = 2x quadrature decoding.  A 600 PPR encoder
  // yields 1200 counts/rev — set encoder_cpr=1200 in mecanum_params.yaml.
  attachInterrupt(digitalPinToInterrupt(FL_ENC_A), isrFL_A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(FR_ENC_A), isrFR_A, CHANGE);
#ifndef USE_BNO055
  // Rear-wheel encoder interrupts share pins with I2C; only attach them
  // when the BNO055 is disabled (bench mode).  See the wiring note above.
  attachInterrupt(digitalPinToInterrupt(RL_ENC_A), isrRL_A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RR_ENC_A), isrRR_A, CHANGE);
#endif
#endif

#ifdef USE_BNO055
  // The Adafruit BNO055 library spins up the Wire bus internally.  begin()
  // can take ~700 ms while the chip self-calibrates its internal NDOF
  // fusion engine — that's fine here, the rest of the system isn't ready
  // yet either.  External crystal mode gives the chip a more stable clock
  // and noticeably reduces fused-orientation drift.
  if (bno.begin()) {
    delay(50);
    bno.setExtCrystalUse(true);
    bnoReady = true;
    Serial.println("BNO055 OK");
  } else {
    bnoReady = false;
    Serial.println("ERR BNO055 NOT DETECTED");
  }
#endif

  stopAllMotors();
  Serial.println("READY");
}

void loop() {
  // ── Parse serial commands (zero-heap static buffer) ─────────────────
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdIdx > 0) {
        cmdBuf[cmdIdx] = '\0';
        processCommand(cmdBuf);
        cmdIdx = 0;
      }
    } else {
      if (cmdIdx < CMD_BUF_SIZE - 1) {
        cmdBuf[cmdIdx++] = c;
      } else {
        // Overflow — discard and reset
        cmdIdx = 0;
      }
    }
  }

  // ── Watchdog: stop motors if no command received ───────────────────
  if (millis() - lastCmdTime > CMD_TIMEOUT_MS) {
    stopAllMotors();
  }

#if defined(USE_ENCODERS) || defined(USE_BNO055)
  // ── Periodic unified telemetry ─────────────────────────────────────
  if (millis() - lastTelemetryReport >= TELEMETRY_REPORT_MS) {
    lastTelemetryReport = millis();
    reportTelemetry();
  }
#endif
}

#if defined(USE_ENCODERS) || defined(USE_BNO055)
void reportTelemetry() {
  long fl = 0, fr = 0, rl = 0, rr = 0;
#ifdef USE_ENCODERS
  noInterrupts();
  fl = encFL; fr = encFR; rl = encRL; rr = encRR;
  interrupts();
#endif

#ifdef USE_BNO055
  if (bnoReady) {
    // BNO055 quaternion is unitless (4 floats).  getQuat() returns the
    // fused orientation in the chip's NDOF/IMU frame; the ROS-side
    // driver re-stamps these with the Pi clock so all consumers see a
    // single timeline.
    imu::Quaternion q  = bno.getQuat();
    imu::Vector<3> g   = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);     // deg/s
    imu::Vector<3> a   = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER); // m/s^2

    Serial.print(F("ODOM "));
    Serial.print(fl); Serial.print(' ');
    Serial.print(fr); Serial.print(' ');
    Serial.print(rl); Serial.print(' ');
    Serial.print(rr); Serial.print(' ');
    // Quaternion (w, x, y, z) — 4 decimals is more than enough for 16-bit
    // fixed-point fusion output.
    Serial.print(q.w(), 4); Serial.print(' ');
    Serial.print(q.x(), 4); Serial.print(' ');
    Serial.print(q.y(), 4); Serial.print(' ');
    Serial.print(q.z(), 4); Serial.print(' ');
    // Gyro deg/s
    Serial.print(g.x(), 3); Serial.print(' ');
    Serial.print(g.y(), 3); Serial.print(' ');
    Serial.print(g.z(), 3); Serial.print(' ');
    // Linear acceleration m/s^2 (includes gravity)
    Serial.print(a.x(), 3); Serial.print(' ');
    Serial.print(a.y(), 3); Serial.print(' ');
    Serial.println(a.z(), 3);
    return;
  }
#endif

#ifdef USE_ENCODERS
  // Fallback when the BNO055 is disabled or failed to initialise — emit
  // the legacy encoder-only packet so wheel-odom-only tooling still works.
  Serial.print(F("E "));
  Serial.print(fl); Serial.print(' ');
  Serial.print(fr); Serial.print(' ');
  Serial.print(rl); Serial.print(' ');
  Serial.println(rr);
#endif
}
#endif

void processCommand(const char* cmd) {
  // Skip leading whitespace (replaces String.trim())
  while (*cmd == ' ' || *cmd == '\t') cmd++;

  if (cmd[0] == 'S') {
    stopAllMotors();
    Serial.println("STOPPED");
    lastCmdTime = millis();
    return;
  }

  if (cmd[0] == 'M') {
    int fl, fr, rl, rr;
    int parsed = sscanf(cmd, "M %d %d %d %d", &fl, &fr, &rl, &rr);

    if (parsed == 4) {
      fl = constrain(fl, -255, 255);
      fr = constrain(fr, -255, 255);
      rl = constrain(rl, -255, 255);
      rr = constrain(rr, -255, 255);

      setMotor(FL_RPWM, FL_LPWM, fl);
      setMotor(FR_RPWM, FR_LPWM, fr);
      setMotor(RL_RPWM, RL_LPWM, rl);
      setMotor(RR_RPWM, RR_LPWM, rr);

      lastCmdTime = millis();

      Serial.print("OK ");
      Serial.print(fl); Serial.print(" ");
      Serial.print(fr); Serial.print(" ");
      Serial.print(rl); Serial.print(" ");
      Serial.println(rr);
    } else {
      Serial.println("ERR PARSE");
    }
    return;
  }

  if (cmd[0] == 'P') {
    Serial.println("PONG");
    return;
  }

  Serial.println("ERR UNKNOWN");
}

/*
 * BTS7960 motor control:
 *   speed > 0  → forward: RPWM = speed, LPWM = 0
 *   speed < 0  → reverse: RPWM = 0,     LPWM = |speed|
 *   speed == 0 → brake:   RPWM = 0,     LPWM = 0
 */
void setMotor(int rpwmPin, int lpwmPin, int speed) {
  if (speed > 0) {
    analogWrite(rpwmPin, speed);
    analogWrite(lpwmPin, 0);
  } else if (speed < 0) {
    analogWrite(rpwmPin, 0);
    analogWrite(lpwmPin, -speed);
  } else {
    analogWrite(rpwmPin, 0);
    analogWrite(lpwmPin, 0);
  }
}

void stopAllMotors() {
  setMotor(FL_RPWM, FL_LPWM, 0);
  setMotor(FR_RPWM, FR_LPWM, 0);
  setMotor(RL_RPWM, RL_LPWM, 0);
  setMotor(RR_RPWM, RR_LPWM, 0);
}

// ── Encoder ISRs — 2x quadrature decoding (CHANGE on Channel A) ─────
// On each Channel-A edge, read Channel B to determine direction.
// CHANGE doubles the count vs RISING: 600 PPR → 1200 counts/rev.
#ifdef USE_ENCODERS
void isrFL_A() { encFL += digitalRead(FL_ENC_B) ? -1 : 1; }
void isrFR_A() { encFR += digitalRead(FR_ENC_B) ? -1 : 1; }
#ifndef USE_BNO055
void isrRL_A() { encRL += digitalRead(RL_ENC_B) ? -1 : 1; }
void isrRR_A() { encRR += digitalRead(RR_ENC_B) ? -1 : 1; }
#endif
#endif
