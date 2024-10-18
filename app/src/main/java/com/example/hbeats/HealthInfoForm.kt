package com.example.hbeats

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.Text
import androidx.compose.material.TextFieldDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp


@Composable
fun HealthInfoForm(
    age: String,
    onAgeChange: (String) -> Unit,
    cholesterol: String,
    onCholesterolChange: (String) -> Unit,
    diabetes: String,
    onDiabetesChange: (String) -> Unit,
    familyHistory: String,
    onFamilyHistoryChange: (String) -> Unit,
    smoking: String,
    onSmokingChange: (String) -> Unit,
    obesity: String,
    onObesityChange: (String) -> Unit,
    alcoholConsumption: String,
    onAlcoholConsumptionChange: (String) -> Unit,
    exerciseHoursPerWeek: String,
    onExerciseHoursPerWeekChange: (String) -> Unit,
    diet: String,
    onDietChange: (String) -> Unit,
    previousHeartProblems: String,
    onPreviousHeartProblemsChange: (String) -> Unit,
    medicationUse: String,
    onMedicationUseChange: (String) -> Unit,
    stressLevel: String,
    onStressLevelChange: (String) -> Unit,
    sedentaryHoursPerDay: String,
    onSedentaryHoursPerDayChange: (String) -> Unit,
    bmi: String,
    onBmiChange: (String) -> Unit,
    triglycerides: String,
    onTriglyceridesChange: (String) -> Unit,
    physicalActivityDaysPerWeek: String,
    onPhysicalActivityDaysPerWeekChange: (String) -> Unit,
    sleepHoursPerDay: String,
    onSleepHoursPerDayChange: (String) -> Unit,
    isMale: String,
    onIsMaleChange: (String) -> Unit,
    systolicPressure: String,
    onSystolicPressureChange: (String) -> Unit,
    diastolicPressure: String,
    onDiastolicPressureChange: (String) -> Unit,
) {
    Column(modifier = Modifier.padding(16.dp)) {
        TextFieldWithLabel(label = "Age", value = age, onValueChange = onAgeChange)
        TextFieldWithLabel(
            label = "Cholesterol",
            value = cholesterol,
            onValueChange = onCholesterolChange
        )
        TextFieldWithLabel(label = "Diabetes", value = diabetes, onValueChange = onDiabetesChange)
        TextFieldWithLabel(
            label = "Family History",
            value = familyHistory,
            onValueChange = onFamilyHistoryChange
        )
        TextFieldWithLabel(label = "Smoking", value = smoking, onValueChange = onSmokingChange)
        TextFieldWithLabel(label = "Obesity", value = obesity, onValueChange = onObesityChange)
        TextFieldWithLabel(
            label = "Alcohol Consumption",
            value = alcoholConsumption,
            onValueChange = onAlcoholConsumptionChange
        )
        TextFieldWithLabel(
            label = "Exercise Hours Per Week",
            value = exerciseHoursPerWeek,
            onValueChange = onExerciseHoursPerWeekChange
        )
        TextFieldWithLabel(label = "Diet", value = diet, onValueChange = onDietChange)
        TextFieldWithLabel(
            label = "Previous Heart Problems",
            value = previousHeartProblems,
            onValueChange = onPreviousHeartProblemsChange
        )
        TextFieldWithLabel(
            label = "Medication Use",
            value = medicationUse,
            onValueChange = onMedicationUseChange
        )
        TextFieldWithLabel(
            label = "Stress Level",
            value = stressLevel,
            onValueChange = onStressLevelChange
        )
        TextFieldWithLabel(
            label = "Sedentary Hours Per Day",
            value = sedentaryHoursPerDay,
            onValueChange = onSedentaryHoursPerDayChange
        )
        TextFieldWithLabel(label = "BMI", value = bmi, onValueChange = onBmiChange)
        TextFieldWithLabel(
            label = "Triglycerides",
            value = triglycerides,
            onValueChange = onTriglyceridesChange
        )
        TextFieldWithLabel(
            label = "Physical Activity Days Per Week",
            value = physicalActivityDaysPerWeek,
            onValueChange = onPhysicalActivityDaysPerWeekChange
        )
        TextFieldWithLabel(
            label = "Sleep Hours Per Day",
            value = sleepHoursPerDay,
            onValueChange = onSleepHoursPerDayChange
        )
        TextFieldWithLabel(label = "Is Male", value = isMale, onValueChange = onIsMaleChange)
        TextFieldWithLabel(
            label = "Systolic Pressure",
            value = systolicPressure,
            onValueChange = onSystolicPressureChange
        )
        TextFieldWithLabel(
            label = "Diastolic Pressure",
            value = diastolicPressure,
            onValueChange = onDiastolicPressureChange
        )
    }
}

@Composable
fun TextFieldWithLabel(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label, color = Color.Black) },
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        colors = TextFieldDefaults.outlinedTextFieldColors(
            focusedBorderColor = Color.Black,
            unfocusedBorderColor = Color.Gray,
            focusedLabelColor = Color.Black
        ),
        shape = MaterialTheme.shapes.medium
    )
}
