package main

import (
	"bytes"
	_ "encoding/json"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/chai2010/webp"
	"github.com/nfnt/resize"
	"gocv.io/x/gocv"
)

func resizeImage(img image.Image, maxWidth, maxHeight uint) image.Image {
	return resize.Thumbnail(maxWidth, maxHeight, img, resize.Lanczos3)
}

func saveAsWebP(img image.Image, outputPath string) error {
	var buf bytes.Buffer
	if err := webp.Encode(&buf, img, &webp.Options{Lossless: true}); err != nil {
		return fmt.Errorf("failed to encode image to WebP: %v", err)
	}
	return ioutil.WriteFile(outputPath, buf.Bytes(), 0644)
}

func saveMatAsWebP(mat gocv.Mat, outputPath string) error {
	img, err := mat.ToImage()
	if err != nil {
		return fmt.Errorf("failed to convert Mat to Image: %v", err)
	}
	return saveAsWebP(img, outputPath)
}

func cropAndSaveFace(img gocv.Mat, face image.Rectangle, index int, outputDir, baseFilename string) error {
	extraWidth := face.Dx() / 2
	extraHeightTop := face.Dy() / 2
	extraHeightBottom := face.Dy()
	cropRect := image.Rect(
		max(0, face.Min.X-extraWidth),
		max(0, face.Min.Y-extraHeightTop),
		min(img.Cols(), face.Max.X+extraWidth),
		min(img.Rows(), face.Max.Y+extraHeightBottom),
	)

	croppedImg := img.Region(cropRect)
	defer croppedImg.Close()

	outputPath := filepath.Join(outputDir, fmt.Sprintf("%s_face_%d.webp", baseFilename, index))
	return saveMatAsWebP(croppedImg, outputPath)
}

func detectFace(imagePath string, outputImagePath string, outputDir string, maxWidth, maxHeight uint) (bool, error) {
	classifier := gocv.NewCascadeClassifier()
	if !classifier.Load("haarcascade_frontalface_default.xml") {
		return false, fmt.Errorf("error loading Haar cascade file")
	}
	defer classifier.Close()

	img := gocv.IMRead(imagePath, gocv.IMReadColor)
	if img.Empty() {
		return false, fmt.Errorf("error reading image")
	}
	defer img.Close()

	resizedImg := gocv.NewMat()
	defer resizedImg.Close()
	gocv.Resize(img, &resizedImg, image.Point{X: int(maxWidth), Y: int(maxHeight)}, 0, 0, gocv.InterpolationLinear)

	grayImg := gocv.NewMat()
	defer grayImg.Close()
	gocv.CvtColor(resizedImg, &grayImg, gocv.ColorBGRToGray)

	faces := classifier.DetectMultiScaleWithParams(
		grayImg, 1.1, 5, 0, image.Point{X: 30, Y: 30}, image.Point{},
	)

	if len(faces) == 0 {
		return false, nil
	}

	baseFilename := filepath.Base(imagePath)
	baseFilename = baseFilename[:len(baseFilename)-len(filepath.Ext(baseFilename))]

	for i, face := range faces {
		extraWidth := face.Dx() / 2
		extraHeightTop := face.Dy() / 2
		extraHeightBottom := face.Dy()
		expandedRect := image.Rect(
			max(0, face.Min.X-extraWidth),
			max(0, face.Min.Y-extraHeightTop),
			min(resizedImg.Cols(), face.Max.X+extraWidth),
			min(resizedImg.Rows(), face.Max.Y+extraHeightBottom),
		)
		gocv.Rectangle(&resizedImg, expandedRect, color.RGBA{255, 0, 0, 0}, 3)

		if err := cropAndSaveFace(resizedImg, face, i+1, outputDir, baseFilename); err != nil {
			return false, fmt.Errorf("error saving face image: %v", err)
		}
	}

	if err := saveMatAsWebP(resizedImg, outputImagePath); err != nil {
		return false, fmt.Errorf("error saving output image in WebP format: %v", err)
	}

	return true, nil
}

func processImages(inputDir, outputDir string, maxWidth, maxHeight uint) error {
	files, err := ioutil.ReadDir(inputDir)
	if err != nil {
		return fmt.Errorf("failed to read input directory: %v", err)
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		inputPath := filepath.Join(inputDir, file.Name())
		outputImagePath := filepath.Join(outputDir, fmt.Sprintf("output_%s.webp", file.Name()))

		fmt.Printf("Processing file: %s\n", inputPath)
		if _, err := detectFace(inputPath, outputImagePath, outputDir, maxWidth, maxHeight); err != nil {
			fmt.Printf("Error processing file %s: %v\n", inputPath, err)
		}
	}

	return nil
}

func main() {
	inputDir := "input_images"
	outputDir := "output_images"
	maxWidth := uint(1024)
	maxHeight := uint(1024)

	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	if err := processImages(inputDir, outputDir, maxWidth, maxHeight); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
