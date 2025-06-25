from fraud_detector import logger
from fraud_detector.config.configuration import ConfigurationManager
from fraud_detector.components.data_transformation import DataTransformation


STAGE_NAME = "Data Ingestion Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(
            config=data_transformation_config)

        # La ruta al archivo de datos de entrada debe venir de la etapa de ingesta
        # Por simplicidad, la tomamos directamente de config.yaml/data_ingestion.local_data_file
        # En un pipeline DVC real, esto lo manejaría DVC
        # Para obtener la ruta del archivo ingestado
        data_ingestion_config = config.get_data_ingestion_config()
        data_transformation.initiate_data_transformation(
            data_ingestion_config.local_data_file)


"""
Python will start reading from here only
"""
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")

    except Exception as e:
        logger.exception(e)
        raise e
