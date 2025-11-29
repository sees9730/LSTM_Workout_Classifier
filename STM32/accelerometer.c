
/* accelerometer.c
 * LSM303DLHC (the on board accelerometer) driver for STM32
 * setting up with 100Hz, ±8g, high-res
 * Daphne Felt - ECEN 5613
 */

#include "accelerometer.h"
#include "stm32f4xx_hal.h"
#include "uart.h"

#define LSM303_ADDR             (0x19 << 1)  // 0x32 after shift

// registers
#define CTRL_REG1_A             0x20
#define CTRL_REG4_A             0x23
#define OUT_X_L_A               0x28

// again, setting to fixed config. These are the hard-coded vals
#define CTRL1_100HZ_ENABLED     0x57 // 100Hz, all axes enabled
#define CTRL4_8G_HIGHRES_BDU    0xA8 // ±8g, high-res, BDU enabled

#define I2C_TIMEOUT             100
static I2C_HandleTypeDef hi2c1;

static void init_i2c(void) {
    __HAL_RCC_I2C1_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    // PB6=SCL, PB9=SDA
    GPIO_InitTypeDef gpio = {0};
    gpio.Pin = GPIO_PIN_6 | GPIO_PIN_9;
    gpio.Mode = GPIO_MODE_AF_OD;
    gpio.Pull = GPIO_PULLUP;
    gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio.Alternate = GPIO_AF4_I2C1;
    HAL_GPIO_Init(GPIOB, &gpio);

    // I2C1, 400kHz
    hi2c1.Instance = I2C1;
    hi2c1.Init.ClockSpeed = 400000;
    hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
    hi2c1.Init.OwnAddress1 = 0;
    hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;

    HAL_I2C_Init(&hi2c1); // built in func they give you
}

static inline void write_reg(uint8_t reg, uint8_t val) {
    HAL_I2C_Mem_Write(&hi2c1, LSM303_ADDR, reg,
                      I2C_MEMADD_SIZE_8BIT, &val, 1, I2C_TIMEOUT);
}

static inline void read_regs(uint8_t reg, uint8_t *buf, uint8_t len) {
    reg |= 0x80;  // auto increment
    HAL_I2C_Mem_Read(&hi2c1, LSM303_ADDR, reg,
                     I2C_MEMADD_SIZE_8BIT, buf, len, I2C_TIMEOUT);
}

bool Accel_Init(void) {
    // init i2c
	sendString("INITIALIZING ACCEL");
    init_i2c();
    HAL_Delay(10);

    // make sure we are connected
    if (HAL_I2C_IsDeviceReady(&hi2c1, LSM303_ADDR, 3, I2C_TIMEOUT) != HAL_OK) {
    	sendString("ACCEL CONNECTION FAILED\n\r");
        return false;
    }

    // config
    write_reg(CTRL_REG1_A, CTRL1_100HZ_ENABLED);
    write_reg(CTRL_REG4_A, CTRL4_8G_HIGHRES_BDU);

    HAL_Delay(10);
    return true;
}

void convert_to_gs(AccelRawData *data){
	data->x = (data->x * 8.0f) / 2048.0f;
	data->y = (data->y * 8.0f) / 2048.0f;
	data->z = (data->z * 8.0f) / 2048.0f;
}

// read all vals
void Accel_ReadRaw(AccelRawData *data) {
    uint8_t buf[6];

    read_regs(OUT_X_L_A, buf, 6); // X_L, X_H, Y_L, Y_H, Z_L, Z_H

    // Combine bytes
    data->x = (int16_t)((buf[1] << 8) | buf[0]) >> 4;
    data->y = (int16_t)((buf[3] << 8) | buf[2]) >> 4;
    data->z = (int16_t)((buf[5] << 8) | buf[4]) >> 4;

    convert_to_gs(data);
}
