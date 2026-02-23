from pathlib import Path
import logging
import logging.config
import yaml
import os

# Flag interno para evitar configurar logging más de una vez
_configured = False

# ============================================================ 
# Configuración del sistema de logging 
# ============================================================
def setup_logging(config_path: str | None = None):
    """Configura el sistema de logging a partir de un archivo YAML."""
    global _configured
    if _configured:
        return

    # Ruta del archivo logging.yaml
    base_dir = Path(__file__).resolve().parent
    cfg = Path(config_path) if config_path else base_dir / "logging.yaml"

    # Crear carpeta logs/ en el root del proyecto
    project_root = Path(__file__).resolve().parents[2] 
    logs_dir = Path(os.environ.get("LOG_DIR", project_root / "logs")) 
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Cargar configuración YAML si existe
    if cfg.exists():
        with open(cfg, "rt", encoding="utf8") as f:
            config = yaml.safe_load(f)

        # Sobrescribir dinámicamente la ruta del archivo de logs
        config["handlers"]["app_file"]["filename"] = str(logs_dir / "app.log")

        # Aplicar configuración
        logging.config.dictConfig(config)

    _configured = True

# ============================================================ 
# Obtener un logger configurado 
# ============================================================
def get_logger(name: str):
    """Obtiene un logger ya configurado."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
