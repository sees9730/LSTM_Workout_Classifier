/* accelerometer.h
 * Simplified LSM303DLHC driver for STM32F411E-DISCO
 * Fixed config: 100Hz, ±8g, high-res
 * Daphne Felt - ECEN 5613
 */

#ifndef ACCELEROMETER_H
#define ACCELEROMETER_H

#include <stdint.h>
#include <stdbool.h>

// Accelerometer data structure (raw values)
typedef struct {
    int16_t x;
    int16_t y;
    int16_t z;
} AccelRawData;

// Function prototypes
bool Accel_Init(void);              // Initialize I2C and accelerometer
void Accel_ReadRaw(AccelRawData *data);  // Read X, Y, Z values

#endif // ACCELEROMETER_H
