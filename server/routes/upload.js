const express = require("express");
const router = express.Router();
const { upload, cloudinary } = require("../config/cloudinary");
const { protect } = require("../middleware/authMiddleware");
const { preprocessImage, applyPrivacyBlur } = require("../services/imageService");
const { checkIsBinPhoto } = require("../services/aiService");

// @route  POST /api/upload
router.post("/", protect, upload.single("photo"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    const originalUrl = req.file.path;

    // Start the bin check now, so it runs alongside preprocessing and blur.
    // In Gemini mode the analysis already checks this, so it is skipped.
    const binCheck =
      process.env.AI_PROVIDER !== "gemini" ? checkIsBinPhoto(originalUrl) : Promise.resolve(true);

    // Apply preprocessing in background (don't block response)
    let processedUrl = originalUrl;
    try {
      const preprocessed = await preprocessImage(originalUrl);
      if (preprocessed.success) {
        processedUrl = preprocessed.processedUrl;
      }
    } catch (e) {
      console.log("Preprocessing skipped:", e.message);
    }

    // Apply privacy blur
    let finalUrl = processedUrl;
    try {
      const privacy = await applyPrivacyBlur(processedUrl);
      if (privacy.success) {
        finalUrl = privacy.privacyUrl;
      }
    } catch (e) {
      console.log("Privacy blur skipped:", e.message);
    }

    // Wait for the bin check, which has been running alongside preprocessing and blur
    const isBin = await binCheck;
    if (isBin === false) {
      await cloudinary.uploader.destroy(req.file.filename).catch(() => {});
      return res.status(400).json({
        message: "This photo does not appear to show a bin. Please upload a clear photo of the bin.",
      });
    }

    res.json({
      url: finalUrl,
      originalUrl,
      publicId: req.file.filename,
      message: "Photo uploaded, preprocessed and privacy-filtered",
    });
  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).json({ message: error.message });
  }
});

router.delete("/:publicId", protect, async (req, res) => {
  try {
    await cloudinary.uploader.destroy(req.params.publicId);
    res.json({ message: "Image deleted" });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

module.exports = router;