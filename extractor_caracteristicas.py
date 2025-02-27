import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import mahotas as mt
from pathlib import Path

# Definir rangos de colores en HSV para análisis de color
color_ranges = {
    'Amarillo': [(20, 100, 100), (30, 255, 255)],
    'Anaranjado': [(10, 100, 100), (20, 255, 255)],
    'Marron': [(10, 50, 50), (20, 200, 150)],
    'Rosado': [(160, 50, 50), (170, 255, 255)],
    'Rojo': [(0, 100, 100), (10, 255, 255)],
    'Morado rojizo': [(140, 50, 50), (160, 255, 255)],
    'Morado': [(130, 50, 50), (140, 255, 255)],
    'Morado violeta': [(125, 50, 50), (135, 255, 255)],
}

# Definir características de formas de tubérculos
shape_descriptors = {
    'Formas del Tubérculo': {
        'Comprimido': lambda ratio: ratio < 1,
        'Esferico': lambda ratio: 0.9 <= ratio <= 1.1,
        'Ovoide': lambda ratio, width_pos: 1 < ratio < 1.5 and width_pos < 1/3,
        'Obovoide': lambda ratio, width_pos: 1 < ratio < 1.5 and width_pos > 2/3,
        'Eliptico': lambda ratio: 1.1 <= ratio <= 1.5,
        'Oblongo': lambda ratio: 1.5 < ratio <= 2,
        'Largo-oblongo': lambda ratio: 1.8 < ratio <= 2.5,
        'Alargado': lambda ratio: ratio > 2.5
    }
}

class PotatoFeatureExtractor:
    def __init__(self, image_path, pixels_to_cm):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.pixels_to_cm = pixels_to_cm
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.filename = os.path.basename(image_path)
        
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        _, self.mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(self.contours) > 0:
            self.contour = max(self.contours, key=cv2.contourArea)
        else:
            raise ValueError("No contours detected in the image")

    def analyze_color(self):
        hsv_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        color_areas = {}
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
            color_areas[color_name] = cv2.countNonZero(mask)
        sorted_colors = sorted(color_areas.items(), key=lambda x: x[1], reverse=True)
        predominant_color = sorted_colors[0][0] if sorted_colors[0][1] > 0 else "Ausente"
        secondary_color = sorted_colors[1][0] if len(sorted_colors) > 1 and sorted_colors[1][1] > 0 else "Ausente"
        return predominant_color, secondary_color

    def analyze_diameters(self):
        rect = cv2.minAreaRect(self.contour)
        width, height = rect[1]
        major_diameter = max(width, height) / self.pixels_to_cm
        minor_diameter = min(width, height) / self.pixels_to_cm
        return major_diameter / 2, minor_diameter / 2

    def shape_features(self):
        rect = cv2.minAreaRect(self.contour)
        width, height = rect[1]
        ratio = height / width if width != 0 else 0
        width_pos = 0.5 if height >= width else 1.0
        
        # Usar shape_descriptors para determinar la forma
        shape_type = "Indefinido"
        for shape, condition in shape_descriptors['Formas del Tubérculo'].items():
            if shape in ["Ovoide", "Obovoide"]:
                if condition(ratio, width_pos):
                    shape_type = shape
                    break
            elif condition(ratio):
                shape_type = shape
                break
        
        # Calcular características adicionales de forma
        area = cv2.contourArea(self.contour)
        perimeter = cv2.arcLength(self.contour, True)
        
        aspect_ratio = width / height if height != 0 else 0
        roundness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        elongation = 1 - roundness

        return {
            'Aspect Ratio': aspect_ratio,
            'Roundness': roundness,
            'Elongation': elongation,
            'Shape Type': shape_type
        }

    def texture_features(self):
        distances = [1]
        angles = [0]
        glcm = graycomatrix(self.gray, distances, angles, 256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        
        haralick = mt.features.haralick(self.gray, return_mean=True)
        
        return {
            'Contrast': contrast,
            'Correlation': correlation,
            'Energy': energy,
            'Entropy': haralick[8]
        }

    def edge_features(self):
        edges_low = cv2.Canny(self.gray, 50, 150)
        edges_high = cv2.Canny(self.gray, 100, 200)
        edges_combined = cv2.bitwise_or(edges_low, edges_high)
        edges_masked = cv2.bitwise_and(edges_combined, self.mask)
        
        total_pixels = np.sum(self.mask > 0)
        edge_density = np.sum(edges_masked > 0) / total_pixels if total_pixels > 0 else 0
        
        return {
            'Edge Density': edge_density
        }

    def hu_moments(self):
        moments = cv2.moments(self.contour)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        
        return {
            'Hu Moment 1': hu_moments[0][0],
            'Hu Moment 2': hu_moments[1][0],
            'Hu Moment 3': hu_moments[2][0]
        }

    def extract_all_features(self):
        try:
            predominant_color, secondary_color = self.analyze_color()
            major_diameter, minor_diameter = self.analyze_diameters()
            
            features = {
                'Predominant Color': predominant_color,
                'Secondary Color': secondary_color,
                'Major Diameter (cm)': major_diameter,
                'Minor Diameter (cm)': minor_diameter,
                **self.shape_features(),
                **self.texture_features(),
                **self.edge_features(),
                **self.hu_moments()
            }
            return features
        except Exception as e:
            print(f"Error extracting features from {self.filename}: {e}")
            return None

def calculate_pixels_to_cm(reference_image_path, known_length_cm):
    img = cv2.imread(reference_image_path)
    if img is None:
        raise ValueError(f"Could not load reference image: {reference_image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours detected in the reference image")
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    reference_length_pixels = max(rect[1])
    return reference_length_pixels / known_length_cm

def process_directory(input_dir, output_csv, pixels_to_cm):
    features_list = []
    for subfolder in Path(input_dir).iterdir():
        if subfolder.is_dir():
            variety = subfolder.name
            for image_path in subfolder.glob('*.[pjg][pn]g'):
                try:
                    extractor = PotatoFeatureExtractor(str(image_path), pixels_to_cm)
                    features = extractor.extract_all_features()
                    if features is not None:
                        features['Variety'] = variety
                        features_list.append(features)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    reference_image_path = r'C:\documentos\semestre 24 II\aprendizaje\papa_referencia.jpg'
    known_length_cm = 10

    input_directory = r'C:\documentos\semestre 24 II\aprendizaje\clasificacion_papas'
    output_csv_file = r'C:\documentos\semestre 24 II\aprendizaje\caracteristicas_papas_parte3.csv'

    try:
        pixels_to_cm = calculate_pixels_to_cm(reference_image_path, known_length_cm)
        process_directory(input_directory, output_csv_file, pixels_to_cm)
    except ValueError as e:
        print(e)
