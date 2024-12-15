import requests
import base64
import re
from pathlib import Path
import os
import fitz
from PIL import Image
import io
from typing import Dict, List
import anthropic
import httpx

class OpenaiVisionProcessor  :
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

    def image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def pdf_page_to_base64(self, pdf_path: str, page_num: int = 0) -> str:
        """Convert a PDF page to base64 string
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to convert (0-based)
            
        Returns:
            str: Base64 encoded string of the PDF page as JPEG
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # Get the page's pixmap (image)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            doc.close()
            return img_base64
            
        except Exception as e:
            raise ValueError(f"Error converting PDF page to base64: {str(e)}")

    def process_image_with_vision(self, image_path: str, query: str) -> dict:
        """Process image with GPT-4o Vision API"""
        try:
            # Check if image_path exists
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found at path: {image_path}")
                
            b64_image = self.image_to_base64(image_path)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
                                Extract and organize all relevant information from this image based on this user query: {query}. 
                                If formatting in the image provides information, indicate as much in your response 
                                (e.g. large text at the top of the image: title: [large text], 
                                tabular data: table: [tabular data], etc...).
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions", 
                headers=headers, 
                json=payload
            )
            response.raise_for_status()
            
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            
            return {
                "status": "success",
                "content": content
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def process_pdf_with_vision(self, pdf_path: str, query: str) -> Dict[str, str]:

        """Process PDF with GPT-4 Vision API by converting pages to images
        
        Args:
            pdf_path: Path to the PDF file
            query: User query for information extraction
            
        Returns:
            Dict[str, str]: Processing result with status and content
        """
        try:
            # Check if PDF exists
            if not os.path.exists(pdf_path):
                raise ValueError(f"PDF file not found at path: {pdf_path}")
                
            doc = fitz.open(pdf_path)
            all_page_content = []
            
            # Process each page
            for page_num in range(len(doc)):
                # Convert page to base64
                b64_page = self.pdf_page_to_base64(pdf_path, page_num)
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }

                payload = {
                    "model": "gpt-4-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"""Extract and organize the relevant informatio from page {page_num + 1} of 
                                    this PDF based on this user query: {query}.
                                    If formatting in the image provides information, indicate as much in your response 
                                    (e.g. large text at the top of the image: title: [large text], 
                                    tabular data: table: [tabular data], etc...).
                                    """
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_page}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 300
                }

                response = requests.post(
                    "https://api.openai.com/v1/chat/completions", 
                    headers=headers, 
                    json=payload
                )
                response.raise_for_status()
                
                page_content = response.json()['choices'][0]['message']['content']
                all_page_content.append(f"[Page {page_num + 1}]\n{page_content}")

            doc.close()
            
            # Combine content from all pages
            combined_content = "\n\n".join(all_page_content)
            
            return {
                "status": "success",
                "content": combined_content
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            } 
        

class AnthropicVisionProcessor  :
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not found")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def pdf_page_to_base64(self, pdf_path: str, page_num: int = 0) -> str:
        """Convert a PDF page to base64 string
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to convert (0-based)
            
        Returns:
            str: Base64 encoded string of the PDF page as JPEG
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # Get the page's pixmap (image)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            doc.close()
            return img_base64
            
        except Exception as e:
            raise ValueError(f"Error converting PDF page to base64: {str(e)}")

    def process_image_with_vision(self, image_path: str, query: str) -> dict:
        """Process image with Claude 3 Vision API"""
        try:
            # Check if image_path exists
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found at path: {image_path}")
                
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Determine media type based on file extension
            media_type = f"image/{Path(image_path).suffix[1:].lower()}"
            if media_type == "image/jpg":
                media_type = "image/jpeg"

            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"""Extract and organize all relevant information from this image based on this user query: {query}. 
                                If formatting in the image provides information, indicate as much in your response 
                                (e.g. large text at the top of the image: title: [large text], 
                                tabular data: table: [tabular data], etc...)."""
                            }
                        ],
                    }
                ],
            )
            
            return {
                "status": "success",
                "content": message.content[0].text
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def process_pdf_with_vision(self, pdf_path: str, query: str) -> Dict[str, str]:
        """Process PDF with Claude 3 Vision API by converting pages to images
        
        Args:
            pdf_path: Path to the PDF file
            query: User query for information extraction
            
        Returns:
            Dict[str, str]: Processing result with status and content
        """
        try:
            # Check if PDF exists
            if not os.path.exists(pdf_path):
                raise ValueError(f"PDF file not found at path: {pdf_path}")
                
            doc = fitz.open(pdf_path)
            all_page_content = []
            
            # Process each page
            for page_num in range(len(doc)):
                # Convert page to base64
                b64_page = self.pdf_page_to_base64(pdf_path, page_num)
                
                message = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": b64_page,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": f"""Extract and organize the relevant information from page {page_num + 1} of 
                                    this PDF based on this user query: {query}.
                                    If formatting in the image provides information, indicate as much in your response 
                                    (e.g. large text at the top of the image: title: [large text], 
                                    tabular data: table: [tabular data], etc...)."""
                                }
                            ],
                        }
                    ],
                )
                
                page_content = message.content[0].text
                all_page_content.append(f"[Page {page_num + 1}]\n{page_content}")

            doc.close()
            
            # Combine content from all pages
            combined_content = "\n\n".join(all_page_content)
            
            return {
                "status": "success",
                "content": combined_content
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
        
