/* main.c - Smartwatch with X-CUBE-AI Workout Detection
 * Daphne Felt - ECEN 5613
 */

#include <stdbool.h>
#include <stdio.h>
#include "stm32f411xe.h"
#include "main.h"
#include "led.h"
#include "spi.h"
#include "lcd.h"
#include "joystick.h"
#include "uart.h"
#include "pages.h"
#include "workout_inference.h"

// Accelerometer definitions (LIS3DSH on SPI1)
#define LIS3DSH_WHO_AM_I    0x0F
#define LIS3DSH_CTRL_REG4   0x20
#define LIS3DSH_OUT_X_L     0x28
#define LIS3DSH_OUT_X_H     0x29
#define LIS3DSH_OUT_Y_L     0x2A
#define LIS3DSH_OUT_Y_H     0x2B
#define LIS3DSH_OUT_Z_L     0x2C
#define LIS3DSH_OUT_Z_H     0x2D
#define ACCEL_CS_PIN        GPIO_PIN_0
#define ACCEL_CS_PORT       GPIOE

// ADC center and deadzone for joystick
#define ADC_CENTER    2028
#define ADC_DEADZONE   400
#define ADC_LOW   2
#define ADC_HIGH  4095

// Simple delay
void delay(volatile uint32_t t) {
    while(t--);
}

// Accelerometer functions
uint8_t Accel_ReadReg(uint8_t reg) {
    uint8_t value = 0;
    reg |= 0x80;  // Set read bit
    
    HAL_GPIO_WritePin(ACCEL_CS_PORT, ACCEL_CS_PIN, GPIO_PIN_RESET);
    SPI1_WriteByte(reg);
    SPI1_WriteByte(0x00);  // Dummy byte
    value = SPI1->DR;
    HAL_GPIO_WritePin(ACCEL_CS_PORT, ACCEL_CS_PIN, GPIO_PIN_SET);
    
    return value;
}

void Accel_WriteReg(uint8_t reg, uint8_t value) {
    reg &= 0x7F;  // Clear read bit
    
    HAL_GPIO_WritePin(ACCEL_CS_PORT, ACCEL_CS_PIN, GPIO_PIN_RESET);
    SPI1_WriteByte(reg);
    SPI1_WriteByte(value);
    HAL_GPIO_WritePin(ACCEL_CS_PORT, ACCEL_CS_PIN, GPIO_PIN_SET);
}

void Accel_Init(void) {
    // Enable GPIOE clock for CS pin
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOEEN;
    
    // Configure CS pin (PE0)
    HAL_GPIO_WritePin(ACCEL_CS_PORT, ACCEL_CS_PIN, GPIO_PIN_SET);
    GPIOE->MODER &= ~(3 << (0*2));
    GPIOE->MODER |= (1 << (0*2));  // Output mode
    GPIOE->OTYPER &= ~(1 << 0);    // Push-pull
    GPIOE->OSPEEDR |= (3 << (0*2)); // High speed
    
    delay(100000);
    
    // Check WHO_AM_I
    uint8_t whoami = Accel_ReadReg(LIS3DSH_WHO_AM_I);
    if (whoami == 0x3F) {
        // Configure: ODR=100Hz, Enable all axes
        Accel_WriteReg(LIS3DSH_CTRL_REG4, 0x67);
    }
}

void Accel_ReadXYZ(int16_t *x, int16_t *y, int16_t *z) {
    uint8_t xl = Accel_ReadReg(LIS3DSH_OUT_X_L);
    uint8_t xh = Accel_ReadReg(LIS3DSH_OUT_X_H);
    *x = (int16_t)((xh << 8) | xl);
    
    uint8_t yl = Accel_ReadReg(LIS3DSH_OUT_Y_L);
    uint8_t yh = Accel_ReadReg(LIS3DSH_OUT_Y_H);
    *y = (int16_t)((yh << 8) | yl);
    
    uint8_t zl = Accel_ReadReg(LIS3DSH_OUT_Z_L);
    uint8_t zh = Accel_ReadReg(LIS3DSH_OUT_Z_H);
    *z = (int16_t)((zh << 8) | zl);
}

// System clock config
void SystemClock_Config(void) {
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM = 8;
    RCC_OscInitStruct.PLL.PLLN = 192;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
    RCC_OscInitStruct.PLL.PLLQ = 8;
    HAL_RCC_OscConfig(&RCC_OscInitStruct);

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                                |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;
    HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3);
}

void Error_Handler(void) {
    __disable_irq();
    while(1) {}
}

void EXTI0_IRQHandler(void) {
    if (EXTI->PR & (1 << 0)) {
        EXTI->PR |= (1 << 0);
    }
}

int main(void) {
    SystemClock_Config();

    /* Enable GPIOD clock (LEDs) */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIODEN;
    GPIOD->MODER |= (1<<(12*2)) | (1<<(13*2)) | (1<<(14*2)) | (1<<(15*2));
    LED_OFF(GPIOD, LED_GREEN);
    LED_OFF(GPIOD, LED_ORANGE);
    LED_OFF(GPIOD, LED_RED);
    LED_OFF(GPIOD, LED_BLUE);

    /* Initialize peripherals */
    SPI1_Init();
    LCD_Init();
    UART_Init();
    Joystick_Init();
    Accel_Init();
    
    /* Initialize X-CUBE-AI workout detection */
    sendStringGreen("Initializing X-CUBE-AI...\r\n");
    if (!Workout_Init()) {
        sendStringGreen("ERROR: X-CUBE-AI initialization failed!\r\n");
        LCD_DrawString(10, 100, "AI INIT FAIL", COLOR_RED, COLOR_BLACK, 2);
        while(1) {
            LED_ON(GPIOD, LED_RED);
            delay(1000000);
            LED_OFF(GPIOD, LED_RED);
            delay(1000000);
        }
    }
    sendStringGreen("X-CUBE-AI initialized successfully!\r\n");

    /* Setup LCD */
    LCD_Clear(COLOR_BLACK);
    LCD_DrawString(10, 10, "SMARTWATCH v2", COLOR_CYAN, COLOR_BLACK, 3);
    LCD_DrawString(10, 50, "AI Workout Detect", COLOR_WHITE, COLOR_BLACK, 2);
    LCD_DrawRect(0, 0, ILI9341_WIDTH-1, ILI9341_HEIGHT-1, COLOR_MAGENTA);

    sendStringGreen("\n=== SMARTWATCH WITH X-CUBE-AI ===\n\r");
    sendStringGreen("Collecting data at 100Hz...\n\r");
    sendStringGreen("Inference runs when HR > 80\n\r\n");

    uint16_t joy_x, joy_y;
    uint8_t text_num = 0;
    uint32_t accel_timer = 0;
    uint32_t last_inference = 0;
    uint32_t status_timer = 0;
    int16_t acc_x, acc_y, acc_z;

    while(1) {
        uint32_t now = HAL_GetTick();
        
        // Sample accelerometer at 100Hz (every 10ms)
        if (now - accel_timer >= 10) {
            accel_timer = now;
            
            // Read accelerometer
            Accel_ReadXYZ(&acc_x, &acc_y, &acc_z);
            
            // Add to workout buffer
            Workout_AddSample(acc_x, acc_y, acc_z);
            
            // Check if we should run inference
            if (Workout_ShouldInfer() && (now - last_inference > 3000)) {
                
                // Visual indicator - blink blue LED during inference
                LED_ON(GPIOD, LED_BLUE);
                
                WorkoutResult result;
                if (Workout_RunInference(&result)) {
                    
                    LED_OFF(GPIOD, LED_BLUE);
                    
                    // Display on UART with all class scores
                    char buf[120];
                    sprintf(buf, "\n>>> WORKOUT DETECTED: %s\r\n", 
                           Workout_GetName(result.predicted_class));
                    sendStringGreen(buf);
                    
                    sprintf(buf, "    Confidence: %.1f%% | Inference: %lums | HR: %d\r\n",
                           result.confidence, result.inference_time_ms, Workout_GetHR());
                    sendString(buf);
                    
                    // Show all class scores
                    sendString("    All scores: ");
                    for (int i = 0; i < NUM_CLASSES; i++) {
                        sprintf(buf, "%s:%.0f%% ", Workout_GetName((WorkoutClass)i), 
                               result.class_scores[i]);
                        sendString(buf);
                    }
                    sendString("\r\n\n");
                    
                    // Display on LCD (bottom section)
                    LCD_FillRect(10, 200, 220, 35, COLOR_BLACK);
                    
                    // Workout name in large text
                    sprintf(buf, "%s", Workout_GetName(result.predicted_class));
                    LCD_DrawString(15, 205, buf, COLOR_CYAN, COLOR_BLACK, 2);
                    
                    // Confidence below
                    sprintf(buf, "%.0f%% (%lums)", result.confidence, result.inference_time_ms);
                    LCD_DrawString(15, 225, buf, COLOR_GREEN, COLOR_BLACK, 1);
                    
                    last_inference = now;
                    
                } else {
                    LED_OFF(GPIOD, LED_BLUE);
                    sendStringGreen("Inference failed!\r\n");
                }
            }
            
            // Show buffer fill status every second
            if (now - status_timer >= 1000) {
                status_timer = now;
                
                uint32_t fill = Workout_GetBufferFillLevel();
                if (fill < 100) {
                    // Still collecting data
                    char buf[60];
                    sprintf(buf, "Collecting: %lu%% | HR: %d     \r", fill, Workout_GetHR());
                    sendString(buf);
                    
                    // Show progress bar on LCD
                    LCD_FillRect(10, 180, 220, 10, COLOR_BLACK);
                    LCD_DrawRect(10, 180, 220, 10, COLOR_WHITE);
                    LCD_FillRect(12, 182, (fill * 216) / 100, 6, COLOR_GREEN);
                    
                } else {
                    // Buffer full, waiting for HR threshold
                    if (Workout_GetHR() <= HR_THRESHOLD) {
                        char buf[60];
                        sprintf(buf, "Ready | HR: %d (waiting >%d)  \r", 
                               Workout_GetHR(), HR_THRESHOLD);
                        sendString(buf);
                    }
                }
            }
        }
        
        // Handle incoming texts
        UART_ProcessInput();
        if (UART_TextReady()) {
            text_num++;
            char *text = UART_GetLine();
            
            // Page navigation commands
            if (strcmp(text,"clock")==0) {
                Pages_Display(PAGE_CLOCK);
            } else if (strcmp(text,"stats")==0) {
                Pages_Display(PAGE_STATS);
            } else if (strcmp(text,"texts")==0) {
                Pages_Display(PAGE_TEXTS);
            } else if (strcmp(text,"reset")==0) {
                Workout_ResetBuffer();
                sendStringGreen("Buffer reset!\r\n");
            }
        }

        // Read joystick for page navigation
        Joystick_Read(&joy_x, &joy_y);
        
        // Joystick LED control (optional - can remove if not needed)
        if (joy_x < ADC_LOW)   LED_ON(GPIOD, LED_GREEN);
        else                   LED_OFF(GPIOD, LED_GREEN);
        
        if (joy_x > ADC_HIGH)  LED_ON(GPIOD, LED_RED);
        else                   LED_OFF(GPIOD, LED_RED);
    }
}
