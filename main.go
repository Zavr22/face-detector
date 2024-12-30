package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	_ "image/draw"
	_ "image/jpeg"
	_ "image/png"
	"io/ioutil"
	"os"

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

func cropAndSaveFace(img gocv.Mat, face image.Rectangle, outputImagePath string, index int) error {
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

	return saveMatAsWebP(croppedImg, fmt.Sprintf("face_%d.webp", index))
}

func detectFace(imagePath string, outputImagePath string, maxWidth, maxHeight uint) (bool, error) {
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

		if err := cropAndSaveFace(resizedImg, face, outputImagePath, i+1); err != nil {
			return false, fmt.Errorf("error saving face image: %v", err)
		}
	}

	if err := saveMatAsWebP(resizedImg, outputImagePath); err != nil {
		return false, fmt.Errorf("error saving output image in WebP format: %v", err)
	}

	return true, nil
}

func processImage(inputPath, outputPath string, maxWidth, maxHeight uint) error {
	hasFace, err := detectFace(inputPath, outputPath, maxWidth, maxHeight)
	if err != nil {
		return fmt.Errorf("failed to detect face: %v", err)
	}

	logData := map[string]interface{}{
		"input_path":  inputPath,
		"output_path": outputPath,
		"has_face":    hasFace,
	}
	logJSON, _ := json.MarshalIndent(logData, "", "  ")
	fmt.Printf("Face detection log:\n%s\n", string(logJSON))

	fmt.Printf("Image successfully processed and saved to: %s\n", outputPath)
	return nil
}

func main() {
	inputPath := "teest.jpg"
	outputPath := "output.webp"
	maxWidth := uint(1024)
	maxHeight := uint(1024)

	if err := processImage(inputPath, outputPath, maxWidth, maxHeight); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
