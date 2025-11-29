"""Módulo para perfiles de llenado rápido de pacientes."""

from typing import Dict, List, Any


def get_profile(profile_name: str) -> Dict[str, Any]:
    """Retorna los valores de un perfil clínico específico.
    
    Args:
        profile_name: Nombre del perfil ('sano', 'factores_riesgo', 'comorbilidades').
        
    Returns:
        Diccionario con los valores del perfil para todas las variables del modelo NHANES.
        
    Raises:
        ValueError: Si el nombre del perfil no es válido.
    """
    profiles = {
        'sano': _get_paciente_sano(),
        'factores_riesgo': _get_paciente_factores_riesgo(),
        'comorbilidades': _get_paciente_comorbilidades()
    }
    
    profile_name_lower = profile_name.lower().replace(' ', '_')
    
    if profile_name_lower not in profiles:
        available = ', '.join(profiles.keys())
        raise ValueError(
            f"Perfil '{profile_name}' no encontrado. "
            f"Perfiles disponibles: {available}"
        )
    
    return profiles[profile_name_lower]


def get_available_profiles() -> List[str]:
    """Retorna la lista de perfiles disponibles.
    
    Returns:
        Lista con los nombres de los perfiles disponibles.
    """
    return ['sano', 'factores_riesgo', 'comorbilidades']


def _get_paciente_sano() -> Dict[str, Any]:
    """Perfil de paciente sano: valores normales/óptimos sin comorbilidades.
    
    Returns:
        Diccionario con valores del perfil de paciente sano.
    """
    return {
        # Biomédicas
        'sleep time': 7.0,  # 7 horas de sueño
        'Minutes sedentary activity': 300.0,  # Actividad sedentaria moderada
        'Waist Circumference': 85.0,  # Circunferencia de cintura normal
        'Systolic blood pressure': 120.0,  # Presión sistólica normal
        'Diastolic blood pressure': 80.0,  # Presión diastólica normal
        'High-density lipoprotein': 1.5,  # HDL normal-alto
        'Triglyceride': 1.2,  # Triglicéridos normales
        'Low-density lipoprotein': 2.5,  # LDL normal
        'Fasting Glucose': 5.5,  # Glucosa en ayunas normal
        'Glycohemoglobin': 5.0,  # Hemoglobina glicosilada normal
        'Body Mass Index': 2,  # BMI normal (1=bajo, 2=normal, 3=sobrepeso, 4=obeso)
        
        # Dietéticas
        'energy': 2000.0,  # Energía calórica normal
        'protein': 70.0,  # Proteína normal
        'Carbohydrate': 250.0,  # Carbohidratos normales
        'Dietary fiber': 25.0,  # Fibra dietética adecuada
        'Total saturated fatty acids': 20.0,  # Ácidos grasos saturados normales
        'Total monounsaturated fatty acids': 25.0,  # Ácidos grasos monoinsaturados normales
        'Total polyunsaturated fatty acids': 15.0,  # Ácidos grasos poliinsaturados normales
        'Potassium': 3000.0,  # Potasio normal
        'Sodium': 2500.0,  # Sodio moderado
        
        # Demográficas
        'gender': 1,  # Masculino (1) o Femenino (2) - usando 1 como default
        'age': 35,  # Edad adulta joven (será normalizada a rango 0-3 automáticamente)
        'Race': 3,  # Raza (valores: 1-5)
        'Marital status': 1,  # Estado civil (valores: 1-6)
        
        # Estilo de vida
        'alcohol': 0,  # No consume alcohol (sin espacio)
        'alcohol ': 0,  # No consume alcohol (con espacio, como en splits)
        'smoke': 0,  # No fuma
        'sleep disorder': 1,  # Sin trastorno del sueño (1=no, 2=sí)
        
        # Salud y condiciones
        'Health Insurance': 1,  # Tiene seguro de salud
        'General health condition': 2,  # Salud buena (1=excelente, 2=buena, 3=regular, 4=mala)
        'depression': 1,  # Sin depresión (1=no, 2=sí)
        'diabetes': 0,  # Sin diabetes
        'hypertension': 0,  # Sin hipertensión
        'high cholesterol': 0,  # Sin colesterol alto
        'Coronary Heart Disease': 0  # Sin enfermedad coronaria
    }


def _get_paciente_factores_riesgo() -> Dict[str, Any]:
    """Perfil de paciente con factores de riesgo moderados.
    
    Returns:
        Diccionario con valores del perfil de paciente con factores de riesgo.
    """
    return {
        # Biomédicas
        'sleep time': 6.0,  # Menos horas de sueño
        'Minutes sedentary activity': 600.0,  # Más actividad sedentaria
        'Waist Circumference': 100.0,  # Circunferencia de cintura aumentada
        'Systolic blood pressure': 135.0,  # Presión sistólica elevada
        'Diastolic blood pressure': 85.0,  # Presión diastólica elevada
        'High-density lipoprotein': 1.0,  # HDL bajo
        'Triglyceride': 2.0,  # Triglicéridos elevados
        'Low-density lipoprotein': 3.5,  # LDL elevado
        'Fasting Glucose': 6.5,  # Glucosa en ayunas en límite alto
        'Glycohemoglobin': 5.8,  # Hemoglobina glicosilada en límite
        'Body Mass Index': 3,  # BMI sobrepeso (1=bajo, 2=normal, 3=sobrepeso, 4=obeso)
        
        # Dietéticas
        'energy': 2500.0,  # Mayor ingesta calórica
        'protein': 90.0,  # Mayor proteína
        'Carbohydrate': 300.0,  # Mayor ingesta de carbohidratos
        'Dietary fiber': 15.0,  # Menos fibra
        'Total saturated fatty acids': 30.0,  # Mayor ingesta de grasas saturadas
        'Total monounsaturated fatty acids': 35.0,  # Mayor ingesta de grasas monoinsaturadas
        'Total polyunsaturated fatty acids': 20.0,  # Mayor ingesta de grasas poliinsaturadas
        'Potassium': 2500.0,  # Potasio normal-bajo
        'Sodium': 3500.0,  # Mayor ingesta de sodio
        
        # Demográficas
        'gender': 1,
        'age': 55,  # Edad adulta media (será normalizada a rango 0-3 automáticamente)
        'Race': 3,
        'Marital status': 1,
        
        # Estilo de vida
        'alcohol': 1,  # Consume alcohol ocasionalmente
        'alcohol ': 1,  # Consume alcohol ocasionalmente (con espacio)
        'smoke': 0,  # No fuma actualmente
        'sleep disorder': 2,  # Trastorno del sueño leve
        
        # Salud y condiciones
        'Health Insurance': 1,
        'General health condition': 3,  # Salud regular
        'depression': 1,
        'diabetes': 0,  # Sin diabetes aún
        'hypertension': 1,  # Con hipertensión
        'high cholesterol': 1,  # Con colesterol alto
        'Coronary Heart Disease': 0  # Sin enfermedad coronaria aún
    }


def _get_paciente_comorbilidades() -> Dict[str, Any]:
    """Perfil de paciente con múltiples comorbilidades.
    
    Returns:
        Diccionario con valores del perfil de paciente con múltiples comorbilidades.
    """
    return {
        # Biomédicas
        'sleep time': 5.5,  # Poco sueño
        'Minutes sedentary activity': 720.0,  # Mucha actividad sedentaria
        'Waist Circumference': 110.0,  # Circunferencia de cintura alta
        'Systolic blood pressure': 150.0,  # Presión sistólica alta
        'Diastolic blood pressure': 90.0,  # Presión diastólica alta
        'High-density lipoprotein': 0.8,  # HDL bajo
        'Triglyceride': 2.8,  # Triglicéridos altos
        'Low-density lipoprotein': 4.2,  # LDL muy alto
        'Fasting Glucose': 8.0,  # Glucosa elevada (diabetes)
        'Glycohemoglobin': 7.5,  # Hemoglobina glicosilada elevada
        'Body Mass Index': 4,  # BMI obeso (1=bajo, 2=normal, 3=sobrepeso, 4=obeso)
        
        # Dietéticas
        'energy': 2200.0,  # Ingesta calórica moderada-alta
        'protein': 80.0,
        'Carbohydrate': 280.0,  # Alta ingesta de carbohidratos
        'Dietary fiber': 12.0,  # Poca fibra
        'Total saturated fatty acids': 35.0,  # Alta ingesta de grasas saturadas
        'Total monounsaturated fatty acids': 40.0,  # Alta ingesta de grasas monoinsaturadas
        'Total polyunsaturated fatty acids': 25.0,  # Alta ingesta de grasas poliinsaturadas
        'Potassium': 2200.0,  # Potasio bajo
        'Sodium': 4000.0,  # Alta ingesta de sodio
        
        # Demográficas
        'gender': 1,
        'age': 70,  # Edad mayor (será normalizada a rango 0-3 automáticamente)
        'Race': 3,
        'Marital status': 1,
        
        # Estilo de vida
        'alcohol': 1,  # Consume alcohol
        'alcohol ': 1,  # Consume alcohol (con espacio)
        'smoke': 1,  # Fumador o ex-fumador
        'sleep disorder': 2,  # Trastorno del sueño
        
        # Salud y condiciones
        'Health Insurance': 2,  # Seguro de salud limitado o sin seguro
        'General health condition': 4,  # Salud mala
        'depression': 2,  # Con depresión
        'diabetes': 1,  # Con diabetes
        'hypertension': 1,  # Con hipertensión
        'high cholesterol': 1,  # Con colesterol alto
        'Coronary Heart Disease': 1  # Con enfermedad coronaria
    }

