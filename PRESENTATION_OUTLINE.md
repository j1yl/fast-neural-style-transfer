
## Slide Deck Outline

### **Slide 1: Title Slide**

* Project Title: *Image-to-Emoji: Stylizing Real-World Images with Neural Style Transfer*
* Your name and course (CS190I S25)
* Date

---

### **Slide 2: Motivation**

* Why emoji?
* Why not just retrieve them?
* Real-world images are expressive—can we make emojis more *visually personalized*?

---

### **Slide 3: Problem Statement**

* What you’re solving:

  > "Translate real-world photos into visual emoji-style art using style transfer."
* Simple examples (image of a dog → emoji dog style)

---

### **Slide 4: Prior Work**

* Emoji retrieval methods (brief mention)
* Neural Style Transfer (Gatys et al.)
* Why stylization is more expressive than retrieval

---

### **Slide 5: Project Overview**

* Input → Content image (real photo)
* Style → Emoji image
* Output → Stylized emoji-themed version
* Diagram or pipeline graphic

---

### **Slide 6: Dataset**

* Content Images: 10 real-world images
* Style Images: Emoji images from [Kaggle dataset](https://www.kaggle.com/datasets/subinium/emojiimage-dataset)
* Preprocessing steps
* Show 2–3 example pairs

---

### **Slide 7: Methodology**

* Fast Neural Style Transfer using PyTorch
* Loss functions: Content loss + style loss
* Pre-trained VGG19 features
* Brief note on training vs inference

---

### **Slide 8: Implementation Details**

* Tools used: PyTorch, torchvision, PIL, etc.
* Model setup and training/inference script
* Screenshot of code or project folder structure

---

### **Slide 9: Stylization Results (Part 1)**

* Show side-by-side examples:

  * Original → Stylized (emoji style 1)

---

### **Slide 10: Stylization Results (Part 2)**

* Show examples with different emoji styles
* Maybe include failure cases or interesting edge results

---

### **Slide 11: Evaluation**

* Qualitative evaluation (visual appeal, emoji resemblance)
* Brief note on human feedback / informal peer review
* Trade-off between content retention and style intensity

---

### **Slide 12: Challenges**

* Limited dataset and time
* Matching emoji style images with clean backgrounds
* Balancing style weight

---

### **Slide 13: Lessons Learned**

* Importance of preprocessing and quality of style images
* Neural style transfer's strengths and limitations
* Visualization is powerful

---

### **Slide 14: Future Work**

* Add more emoji styles and allow user selection
* Use segmentation to target parts of the image
* Real-time stylization (with mobile support?)

---

### **Slide 15: Conclusion**

* What was achieved
* What makes this fun and useful
* One final before/after image to end with a smile

---

### **Slide 16: Q\&A**

* “Thank you!” and invite questions

---

## Timeline
| Time      | Topic                          | Notes                                                                            |
| --------- | ------------------------------ | -------------------------------------------------------------------------------- |
| 0–2 min   | **Intro + Problem Motivation** | What’s the problem? Why convert images into emoji styles?                        |
| 2–4 min   | **Prior Work / Techniques**    | Briefly mention neural style transfer, emoji retrieval vs stylization.           |
| 4–7 min   | **Dataset**                    | Describe your content image selection and emoji dataset (Kaggle). Show examples. |
| 7–12 min  | **Methodology**                | Explain model choice (Fast Neural Style Transfer), training/setup, architecture. |
| 12–16 min | **Results**                    | Show stylized outputs side-by-side. Talk about quality, surprises, and tuning.   |
| 16–18 min | **Challenges + Decisions**     | Small dataset, time limits, tradeoffs. Why stylization over retrieval?           |
| 18–20 min | **Conclusion + Future Work**   | Summarize and suggest next steps (e.g., real-time stylization, emoji animation). |
