import traceback
import sys

# ============================================================ 
# Logging para eventos informativos 
# ============================================================
def log_info(logger, message, *, run_id=None, model_version=None, **extra):
    """Registra un evento informativo con datos adicionales."""
    payload = {}
    if run_id is not None: payload["run_id"] = run_id
    if model_version is not None: payload["model_version"] = model_version

    # Añadir cualquier otro dato extra
    payload.update(extra)

    # Log estructurado
    logger.info(message, extra=payload)

# ============================================================ 
# Logging para errores con traceback completo 
# ============================================================
def log_error(logger, message, *, run_id=None, model_version=None, input_dict=None, **extra):
    """Registra un error con traceback completo y datos adicionales."""
    # Obtener información de la excepción activa
    exc_type, exc_value, exc_tb = sys.exc_info()
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

    payload = {
        "error_type": exc_type.__name__,
        "error_message": str(exc_value),
        "traceback": tb,
    }
    if input_dict is not None: payload["input"] = input_dict
    if run_id is not None: payload["run_id"] = run_id
    if model_version is not None: payload["model_version"] = model_version

    # Añadir cualquier otro dato extra
    payload.update(extra)

    # Log estructurado del error
    logger.error(message, extra=payload)
