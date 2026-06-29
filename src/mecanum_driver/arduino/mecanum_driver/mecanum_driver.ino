// Mecanum driver for Arduino Mega + 4x BTS7960 motor drivers
// Serial protocol: <m1,m2,m3,m4>\n  where each value is -255 to 255
// Positive = forward, Negative = reverse, 0 = stop

// --- Pin definitions ---
// Each BTS7960 uses: RPWM (forward PWM), LPWM (reverse PWM), R_EN, L_EN

// Motor 1 - Front Left
const int M1_RPWM = 2;
const int M1_LPWM = 3;
const int M1_R_EN = 22;
const int M1_L_EN = 23;

// Motor 2 - Front Right
const int M2_RPWM = 4;
const int M2_LPWM = 5;
const int M2_R_EN = 24;
const int M2_L_EN = 25;

// Motor 3 - Rear Left
const int M3_RPWM = 6;
const int M3_LPWM = 7;
const int M3_R_EN = 26;
const int M3_L_EN = 27;

// Motor 4 - Rear Right
const int M4_RPWM = 8;
const int M4_LPWM = 9;
const int M4_R_EN = 28;
const int M4_L_EN = 29;

const int rpwm[] = {M1_RPWM, M2_RPWM, M3_RPWM, M4_RPWM};
const int lpwm[] = {M1_LPWM, M2_LPWM, M3_LPWM, M4_LPWM};
const int r_en[] = {M1_R_EN, M2_R_EN, M3_R_EN, M4_R_EN};
const int l_en[] = {M1_L_EN, M2_L_EN, M3_L_EN, M4_L_EN};

int motorVals[4] = {0, 0, 0, 0};

char inputBuffer[64];
int bufferIndex = 0;
bool receiving = false;

unsigned long lastCmdTime = 0;
const unsigned long TIMEOUT_MS = 1000;

void setup() {
    Serial.begin(115200);

    for (int i = 0; i < 4; i++) {
        pinMode(rpwm[i], OUTPUT);
        pinMode(lpwm[i], OUTPUT);
        pinMode(r_en[i], OUTPUT);
        pinMode(l_en[i], OUTPUT);
        digitalWrite(r_en[i], HIGH);
        digitalWrite(l_en[i], HIGH);
        analogWrite(rpwm[i], 0);
        analogWrite(lpwm[i], 0);
    }

    Serial.println("mecanum_driver ready");
}

void setMotor(int index, int val) {
    val = constrain(val, -255, 255);

    if (val > 0) {
        analogWrite(rpwm[index], val);
        analogWrite(lpwm[index], 0);
    } else if (val < 0) {
        analogWrite(rpwm[index], 0);
        analogWrite(lpwm[index], -val);
    } else {
        analogWrite(rpwm[index], 0);
        analogWrite(lpwm[index], 0);
    }
}

void stopAll() {
    for (int i = 0; i < 4; i++) {
        motorVals[i] = 0;
        setMotor(i, 0);
    }
}

void parseCommand(char* cmd) {
    int vals[4];
    int count = 0;
    char* token = strtok(cmd, ",");

    while (token != NULL && count < 4) {
        vals[count] = atoi(token);
        count++;
        token = strtok(NULL, ",");
    }

    if (count == 4) {
        for (int i = 0; i < 4; i++) {
            motorVals[i] = vals[i];
            setMotor(i, motorVals[i]);
        }
        lastCmdTime = millis();
    }
}

void loop() {
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '<') {
            receiving = true;
            bufferIndex = 0;
        } else if (c == '>' && receiving) {
            inputBuffer[bufferIndex] = '\0';
            parseCommand(inputBuffer);
            receiving = false;
        } else if (receiving && bufferIndex < 63) {
            inputBuffer[bufferIndex++] = c;
        }
    }

    // Safety: stop if no command received within timeout
    if (millis() - lastCmdTime > TIMEOUT_MS && lastCmdTime > 0) {
        stopAll();
        lastCmdTime = 0;
    }
}
