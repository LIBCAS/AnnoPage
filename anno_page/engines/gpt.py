class ChatGPTImageCaptioning:
    def __init__(self, config, config_path):
        self.api_key = config["api_key"]
        self.max_image_size = config.getint('max_image_size', fallback=None)
        self.categories = config["categories"]

    def process_page(self, page_image, page_layout):
        if self.categories is not None:
            regions = []
            images = []

            for region in page_layout.regions:
                if region.category in self.categories:
                    y1 = round(min([point[1] for point in region.polygon]))
                    y2 = round(max([point[1] for point in region.polygon]))
                    x1 = round(min([point[0] for point in region.polygon]))
                    x2 = round(max([point[0] for point in region.polygon]))

                    original_width = x2 - x1
                    original_height = y2 - y1

                    image = page_image[y1:y2, x1:x2]

                    if self.max_image_size is not None and image.size > 0:
                        if original_width > self.max_image_size or original_height > self.max_image_size:
                            if original_width > original_height:
                                image = cv2.resize(image, (self.max_image_size, round(self.max_image_size * original_height / original_width)))
                            else:
                                image = cv2.resize(image, (round(self.max_image_size * original_width / original_height), self.max_image_size))

                    if image.size == 0:
                        print(f"Empty region detected {region.id} ({region.category}): {x1},{y1} {x2},{y2}")

                    else:
                        images.append(image)
                        regions.append(region)

            with Pool(4) as p:
                image_captions = p.map(self.generate_image_caption, images)

            for region, image_caption in zip(regions, image_captions):
                caption_en, caption_cz, topics_en, topics_cz, color_en, color_cz = image_caption.split("|")
                region.metadata = GraphicalObjectMetadata(object_id=region.id,
                                                          caption_en=caption_en.strip(),
                                                          caption_cz=caption_cz.strip(),
                                                          topics_en=topics_en.strip(),
                                                          topics_cz=topics_cz.strip(),
                                                          color_en=color_en.strip(),
                                                          color_cz=color_cz.strip(),
                                                          lines_caption=None,
                                                          lines_reference=None)

        return page_layout

    def generate_image_caption(self, image):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            # "text": "Give me one short sentence describing the image."
                            "text": "For the given image, provide details abount its content in the following format (omit the 'less than' and 'greater than' symbols): <english-caption>|<czech-caption>|<english-topics>|<czech-topics>|<english-color>|<czech-color>. The <english-caption> and <czech-caption> should be a full sentences describing the image in english and czech, respectively. The <english-topics> and <czech-topics> should be comma-separated lists of topics in english and czech, respectively. Select the <english-color> and <czech-color> from the following list according to the appearance of the image in english and czech: [black-and-white, grayscale, duotone, color] and [černobílý, šedotónový, dvojbarevný, barevný]."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        try:
            image_caption = response.json()["choices"][0]["message"]["content"]
        except:
            image_caption = ""

        return image_caption

    @staticmethod
    def encode_image(image):
        image_jpg = cv2.imencode('.jpg', image)[1]
        image_base64 = base64.b64encode(image_jpg).decode('utf-8')
        return image_base64
