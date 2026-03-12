# Synthetic French Exam Generator for placement model IA

Un generador híbrido (LLM-Service-Python) service automático de exámenes de francés con evaluación mediante IA. Genera miles de ejemplos de entrenamiento sintéticos para entrenar modelos de evaluación de idioma francés.

## 🎯 Características

- **Generación Automática**: Genera exámenes sintéticos completos basados en esquema de entrenamiento existente
- **Inyección Inteligente de Errores**: Añade errores típicos de estudiantes de francés (acentos, conjugación, ortografía)
- **Múltiples Niveles CEFR**: Soporte para de A1 a C2 con variación automática de complejidad
- **LLM Flexible**: Soporte para múltiples proveedores de IA:
  - **Groq** (recomendado) - API gratuita, muy rápido
- **Preguntas Abiertas y MCQ**: Genera tanto respuestas abiertas como opción múltiple
- **Reintentos Automáticos**: Manejo de fallos con reintentos configurables
- **Logging Estructurado**: Logs JSON para monitoreo y debugging
- **Escalable**: Soporta generación de 500, 2000, 5000, 10000+ exámenes

## 📋 Requisitos

- Python 3.8+
- Acceso a API Groq (gratuita)
- Conexión a internet (para Groq)

## 🚀 Instalación Rápida

### Opción 1: Con Groq (Recomendado - API Gratuita)

```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno virtual (Windows)
venv\Scripts\activate

# En Linux/Mac:
# source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API key de Groq
# - Ir a https://console.groq.com/keys
# - Crear una cuenta gratuita
# - Copiar tu API key
# - Crear archivo .env en el directorio del proyecto:

cat > .env << EOF
LLM_PROVIDER=groq
GROQ_API_KEY=tu_groq_api_key_aqui
TRAINER_API_URL=http://localhost:8000/train
NUM_EXAMS_TO_GENERATE=10
EOF

# En Windows PowerShell, usar:
# @"
# LLM_PROVIDER=groq
# GROQ_API_KEY=tu_groq_api_key_aqui
# TRAINER_API_URL=http://localhost:8000/train
# NUM_EXAMS_TO_GENERATE=10
# "@ | Out-File -Encoding utf8 .env

# 6. Ejecutar generador
python generator.py
```

## 💻 Uso

### Generar 1 exámen (por defecto)
```bash
python generator.py
```

### Generar número específico de exámenes
```bash
python generator.py --num-exams 100
```

### Ejecutar en modo infinito
```bash
python generator.py --infinite
```

### Generar 5000 exámenes
```bash
python generator.py --num-exams 5000
```

### Cambiar URL de API de entrenamiento
```bash
python generator.py --api-url http://127.0.0.1:8000/train
```

## ⚙️ Configuración

### Variables de Entorno (.env)

```bash
# Groq API (https://console.groq.com/keys)
GROQ_API_KEY=your_key_here

# Trainer API
TRAINER_API_URL=http://localhost:8000/train
TRAINER_API_TIMEOUT=30

# Generación
NUM_EXAMS_TO_GENERATE=1
INFINITE_MODE=true

# Probabilidades
ERROR_INJECTION_PROBABILITY=0.4    # 0-1
MCQ_CORRECT_PROBABILITY=0.70       # 0-1

# Logging
LOG_LEVEL=INFO
LOG_FILE=synthetic_training_generator.log
```

## 🧪 Características Avanzadas

### Inyección de Errores

El generador inyecta automáticamente estos tipos de errores:
- **Acentos**: Eliminación aleatoria de acentos (é→e, à→a, etc.)
- **Conjugación**: Errores de concordancia verbo-sujeto
- **Ortografía**: Errores ortográficos comunes
- **Plurales**: Errores en pluralización

Configurable vía `ERROR_INJECTION_PROBABILITY`.

### Variación por Nivel CEFR

| Nivel | Palabras Min | Palabras Max | Complejidad |
|-------|-------------|-------------|-----------|
| A1    | 10          | 30          | Muy Simple |
| A2    | 30          | 60          | Simple    |
| B1    | 60          | 120         | Intermedia |
| B2    | 100         | 160         | Avanzada  |
| C1    | 150         | 250         | Experto   |
| C2    | 200         | 300         | Mastery   |

### Reintentos Automáticos

- Hasta 3 intentos por examen
- Retraso de 2 segundos entre intentos
- Diferentes estrategias según tipo de error

## 📊 Logging

Los logs se guardan en `synthetic_training_generator.log` en formato JSON con:
- Timestamp
- Nivel (INFO, WARNING, ERROR)
- Exam ID y Candidate ID
- Estado de envío
- Códigos de respuesta HTTP

Ejemplo:
```json
{
  "timestamp": "2026-03-12T10:30:45.123456",
  "level": "INFO",
  "module": "generator",
  "message": "Generated exam exam_abc123",
  "exam_id": "exam_abc123",
  "status": "generated"
}
```

## 🔌 Esquema de Examen

El generador produce exámenes con este esquema:

```json
{
  "exam_id": "exam_abc123",
  "candidate_id": "candidate_12345",
  "adaptive": true,
  "questions": [
    {
      "id": "q_fr_1",
      "text": "Pregunta en francés...",
      "type": "open|mcq",
      "language": "fr",
      "difficulty": 1-3,
      "options": ["opt1", "opt2"] // solo para MCQ
    }
  ],
  "answers": [
    {
      "question_id": "q_fr_1",
      "student_answer": "Respuesta generada...", // para open
      "selected_option": "opt1",                  // para MCQ
      "is_correct": true|false,                   // para MCQ
      "time_spent_sec": 120
    }
  ]
}
```

## 🐛 Troubleshooting

### Error: "GROQ_API_KEY is required"
```bash
# Verifica que tu .env contiene GROQ_API_KEY
cat .env | grep GROQ_API_KEY

# O establece como variable de entorno
export GROQ_API_KEY=tu_clave_aqui
```

### Error: "Cannot connect to API endpoint"
```bash
# Verifica que tu servidor de entrenamiento está corriendo
curl http://localhost:8000/train

# Cambiar URL si es diferente
python generator.py --api-url http://192.168.1.100:8000/train
```


### Logs en blanco o vacíos
```bash
# Aumentar nivel de logging
export LOG_LEVEL=DEBUG
python generator.py
```

## 📈 Escalabilidad

El generador puede manejar:
- **500 exámenes**: ~5-10 minutos (depende de LLM)
- **2000 exámenes**: ~20-40 minutos
- **5000 exámenes**: ~1-2 horas
- **10000+ exámenes**: Modo infinito recomendado

Los tiempos dependen de:
- Velocidad de la API LLM
- Latencia de red
- Velocidad de servidor de entrenamiento

## 🤖 Comparación de Proveedores LLM

Solo se usa **Groq**: Gratis, muy rápido (2 min setup), excelente calidad.

## 📝 Ejemplo: Generar 2000 Exámenes

```bash
# Configurar .env
echo "GROQ_API_KEY=sk-..." > .env
echo "TRAINER_API_URL=http://localhost:8000/train" >> .env
echo "NUM_EXAMS_TO_GENERATE=2000" >> .env
echo "LLM_PROVIDER=groq" >> .env

# Ejecutar
python generator.py

# Monitorear progreso
tail -f synthetic_training_generator.log | grep "exam_"
```

## 📄 Licencia

Este proyecto es parte del sistema de placement test model IA.

## 🤝 Soporte

Para problemas, revisa:
1. El archivo `synthetic_training_generator.log`
2. Los ejemplos en este README
3. La configuración en `config.py`

---

**Creado**: Marzo 2026  
**Versión**: 1.0.0  
**Python**: 3.8+
