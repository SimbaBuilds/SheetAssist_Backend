from fastapi import HTTPException
from typing import Tuple, Any, Dict, List, Optional
from functools import lru_cache
from openai import OpenAI
from anthropic import Anthropic
import logging
from app.schemas import SandboxResult, FileDataInfo
import json
from app.utils.system_prompts import gen_from_query_prompt, gen_from_error_prompt, gen_from_analysis_prompt, analyze_sandbox_prompt, sentiment_analysis_prompt, file_namer_prompt, gen_visualization_prompt
import httpx
from PIL import Image
import io
from typing import Dict, List, Union, Tuple
import httpx
from openai import OpenAI
from app.schemas import FileDataInfo
import time
import base64
from pathlib import Path
import os
import fitz
from dotenv import load_dotenv


# Add at the top of the file, after imports
load_dotenv(override=True)

MAX_SNAPSHOT_LENGTH = int(os.getenv("MAX_SNAPSHOT_LENGTH"))
MAX_VISION_OUTPUT_TOKENS = int(os.getenv("MAX_VISION_OUTPUT_TOKENS"))


def build_input_data_snapshot(input_data: List[FileDataInfo]) -> str:
    input_data_snapshot = ""
    for data in input_data:
        data_snapshot = data.snapshot
        input_data_snapshot += f"Original file name: {data.original_file_name}\nData type: {data.data_type}\nData Snapshot:\n{data_snapshot}\n\n"
    return input_data_snapshot


class OpenaiVisionProcessor  :
    def __init__(self, openai_client: OpenAI = None):

        self.client = openai_client 

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

    def process_image_with_vision(self, image_path: str, query: str, input_data: List[FileDataInfo]) -> dict:
        """Process image with GPT-4o Vision API"""
        try:
            # Check if image_path exists
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found at path: {image_path}")
                
            b64_image = self.image_to_base64(image_path)
            
            input_data_snapshot = build_input_data_snapshot(input_data)
            
            completion = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MAIN_MODEL"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
                                Your job is to extract relevant information from an image based on a user query and input data.
                                Extract only the relevant information from the image based on the query and data.
                                If formatting in the image provides information, indicate as much in your response 
                                (e.g. large text at the top of the image: title: [large text], 
                                tabular data: table: [tabular data], etc...).  Query and input data snapshot below in triple backticks.
                                ```Query: {query} 
                                Input Data Snapshot: 
                                {input_data_snapshot}
                                ```
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
                max_tokens=MAX_VISION_OUTPUT_TOKENS
            )
            print(f"\n ------- LLM called with query: {query} and input data snapshot: {input_data_snapshot} ------- \n")
            return {
                "status": "completed",
                "content": completion.choices[0].message.content
            }

        except httpx.ConnectError as e:
            return {
                "status": "error",
                "error": "Connection error to OpenAI service",
                "detail": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def process_pdf_with_vision(
        self, 
        pdf_path: str, 
        query: str, 
        input_data: List[FileDataInfo],
        page_range: Optional[tuple[int, int]] = None
    ) -> Dict[str, str]:
        """Process PDF with GPT-4o Vision API by converting pages to images"""
        try:
            # Check if PDF exists
            if not os.path.exists(pdf_path):
                raise ValueError(f"PDF file not found at path: {pdf_path}")
                
            doc = fitz.open(pdf_path)
            all_page_content = ""
            
            input_data_snapshot = build_input_data_snapshot(input_data)
            
            # Determine page range
            start_page = int(page_range[0]) if page_range else 0
            end_page = min(int(page_range[1]), len(doc)) if page_range else len(doc)
            
            b64_pages = []
            for page_num in range(start_page, end_page):
                b64_page = self.pdf_page_to_base64(pdf_path, page_num)
                b64_pages.append(b64_page)
            i = 0    
            # Process only pages in range
            for b64_page in b64_pages:
                completion = self.client.chat.completions.create(
                    model=os.getenv("OPENAI_MAIN_MODEL"),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                "type": "text",
                                "text": f"""Your job is to extract relevant information from this pdf image based on a user query and input data.
                                Extract only the relevant information from the image based on the query and data.
                                If formatting in the image provides information, indicate as much in your response 
                                (e.g. large text at the top of the image: title: [large text], 
                                tabular data: table: [tabular data], etc...).
                                Query and input data snapshot below in triple backticks.
                                ```Query: {query} 
                                Input Data Snapshot: 
                                {input_data_snapshot}
                                ```
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
                    max_tokens=MAX_VISION_OUTPUT_TOKENS
                )
                i += 1
                page_content = completion.choices[0].message.content
                print(f"""\n -------LLM called with query: {query} on page: {i} of batch ------- \n\n
                Input data snapshot:\n {input_data_snapshot}
                Page Content:\n {page_content}\n
                """)

                all_page_content += f"[Page {i}]:\n{page_content}\n\n"
                time.sleep(float(os.getenv("SLEEP_TIME")))

            doc.close()
            
            return {
                "status": "completed",
                "content": all_page_content
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


class AnthropicVisionProcessor  :
    def __init__(self, anthropic_client: Anthropic = None):

        self.client = anthropic_client 

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

    def process_image_with_vision(self, image_path: str, query: str, input_data: List[FileDataInfo]) -> dict:
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

            input_data_snapshot = build_input_data_snapshot(input_data)
            
            message = self.client.messages.create(
                model=os.getenv("ANTHROPIC_MAIN_MODEL"),
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
                                "text": f"""Your job is to extract relevant information from an image based on a user query and input data.
                                Extract only the relevant information from the image based on the query and data.
                                If formatting in the image provides information, indicate as much in your response 
                                (e.g. large text at the top of the image: title: [large text], 
                                tabular data: table: [tabular data], etc...).
                                Query and input data snapshot below in triple backticks.
                                ```Query: {query} 
                                Input Data Snapshot: 
                                {input_data_snapshot}
                                ```
                                """
                            }
                        ],
                    }
                ],
            )
            
            return {
                "status": "completed",
                "content": message.content[0].text
            }

        except httpx.ConnectError as e:
            return {
                "status": "error",
                "error": "Connection error to Anthropic service",
                "detail": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def process_pdf_with_vision(
        self, 
        pdf_path: str, 
        query: str, 
        input_data: List[FileDataInfo],
        page_range: Optional[tuple[int, int]] = None
    ) -> Dict[str, str]:
        """Process PDF with Claude 3 Vision API by converting pages to images"""
        try:
            if not os.path.exists(pdf_path):
                raise ValueError(f"PDF file not found at path: {pdf_path}")
                
            doc = fitz.open(pdf_path)
            all_page_content = []
            
            input_data_snapshot = build_input_data_snapshot(input_data)
            
            # Determine page range
            start_page = int(page_range[0]) if page_range else 0
            end_page = min(int(page_range[1]), len(doc)) if page_range else len(doc)
            

            b64_pages = []
            for page_num in range(start_page, end_page):
                b64_page = self.pdf_page_to_base64(pdf_path, page_num)
                b64_pages.append(b64_page)
            i = 0
            # Process only pages in range
            for b64_page in b64_pages:
                i += 1
                message = self.client.messages.create(
                    model=os.getenv("ANTHROPIC_MAIN_MODEL"),
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
                                    "text": f"""Your job is to extract relevant information from this pdf image based on a user query and input data.
                                    Extract only the relevant information from the image based on the query and data.
                                    If formatting in the image provides information, indicate as much in your response 
                                    (e.g. large text at the top of the image: title: [large text], 
                                    tabular data: table: [tabular data], etc...).
                                    Query and input data snapshot below in triple backticks.
                                    ```Query: {query} 
                                    Input Data Snapshot: 
                                    {input_data_snapshot}
                                    ```
                                    """
                                }
                            ],
                        }
                    ],
                )
                
                page_content = message.content[0].text
                print(f"""\n -------LLM called with query: {query} on page: {i} of batch ------- \n\n
                Input data snapshot:\n {input_data_snapshot}
                Page Content:\n {page_content}\n
                """)
                all_page_content.append(f"[Page {i}]\n{page_content}")
                time.sleep(float(os.getenv("SLEEP_TIME")))

            doc.close()
            
            # Combine content from all pages
            combined_content = "\n\n".join(all_page_content)
            
            return {
                "status": "completed",
                "content": combined_content
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
        

class LLMService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self._operation_map = {
            "generate_text": {
                "openai": self._openai_generate_text,
                "anthropic": self._anthropic_generate_text
            },
            "process_image_with_vision": {
                "openai": self._openai_process_image_with_vision,
                "anthropic": self._anthropic_process_image_with_vision
            },
            "process_pdf_with_vision": {
                "openai": self._openai_process_pdf_with_vision,
                "anthropic": self._anthropic_process_pdf_with_vision
            },
            "gen_from_query": {
                "openai": self._openai_gen_from_query,
                "anthropic": self._anthropic_gen_from_query
            },
            "gen_from_error": {
                "openai": self._openai_gen_from_error,
                "anthropic": self._anthropic_gen_from_error
            },
            "gen_from_analysis": {
                "openai": self._openai_gen_from_analysis,
                "anthropic": self._anthropic_gen_from_analysis
            },
            "analyze_sandbox_result": {
                "openai": self._openai_analyze_sandbox_result,
                "anthropic": self._anthropic_analyze_sandbox_result
            },
            "sentiment_analysis": {
                "openai": self._openai_sentiment_analysis,
                "anthropic": self._anthropic_sentiment_analysis
            },
            "file_namer": {
                "openai": self._openai_file_namer,
                "anthropic": self._anthropic_file_namer
            },
            "gen_visualization": {
                "openai": self._openai_gen_visualization,
                "anthropic": self._anthropic_gen_visualization
            }
        }

        # Add system prompts as class attributes
        self._gen_from_query_prompt = gen_from_query_prompt
        self._gen_from_error_prompt = gen_from_error_prompt
        self._gen_from_analysis_prompt = gen_from_analysis_prompt
        self._analyze_sandbox_prompt = analyze_sandbox_prompt
        self._sentiment_analysis_prompt = sentiment_analysis_prompt
        self._file_namer_prompt = file_namer_prompt
        self._gen_visualization_prompt = gen_visualization_prompt


    async def execute_with_fallback(self, operation: str, *args, **kwargs) -> Tuple[str, Any]:
        try:
            # Try OpenAI first
            result = await self._execute_openai(operation, *args, **kwargs)
            return "openai", result
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            # Specifically handle connection errors
            logging.error(f"OpenAI connection error: {str(e)}")
            try:
                # Fallback to Anthropic
                result = await self._execute_anthropic(operation, *args, **kwargs)
                return "anthropic", result
            except (httpx.ConnectError, httpx.ConnectTimeout) as e2:
                # Both providers failed with connection errors
                raise HTTPException(
                    status_code=503,  # Service Unavailable
                    detail="Unable to connect to AI providers"
                )
            except Exception as e2:
                # Other Anthropic errors
                raise HTTPException(
                    status_code=500,
                    detail=f"Both providers failed. OpenAI: Connection Error, Anthropic: {str(e2)}"
                )
        except Exception as e:
            # Handle other OpenAI errors
            logging.warning(f"OpenAI failed: {str(e)}. Falling back to Anthropic.")
            try:
                # Fallback to Anthropic
                result = await self._execute_anthropic(operation, *args, **kwargs)
                return "anthropic", result
            except Exception as e2:
                raise HTTPException(
                    status_code=500,
                    detail=f"Both providers failed. OpenAI: {str(e)}, Anthropic: {str(e2)}"
                )

    async def _execute_openai(self, operation: str, *args, **kwargs):
        if operation not in self._operation_map:
            raise ValueError(f"Unknown operation: {operation}")
        return await self._operation_map[operation]["openai"](*args, **kwargs)

    async def _execute_anthropic(self, operation: str, *args, **kwargs):
        if operation not in self._operation_map:
            raise ValueError(f"Unknown operation: {operation}")
        return await self._operation_map[operation]["anthropic"](*args, **kwargs)

    async def _openai_generate_text(self, system_prompt: str, user_content: str) -> str:
        """Generate text using OpenAI with system and user prompts"""
        response = self.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MAIN_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        if not response.choices[0].message.content:
            raise ValueError("Empty response from API")
        return response.choices[0].message.content

    async def _anthropic_generate_text(self, system_prompt: str, user_content: str) -> str:
        """Generate text using Anthropic with system and user prompts"""
        response = self.anthropic_client.messages.create(
            model=os.getenv("ANTHROPIC_MAIN_MODEL"),
            max_tokens=5000,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": user_content
                        }
                    ]
                }
            ]
        )
        if not response.content[0].text:
            raise ValueError("Empty response from API")
        return response.content[0].text

    async def _openai_process_image_with_vision(
        self, 
        image_path: str, 
        query: str, 
        input_data: List[FileDataInfo]
    ) -> Dict[str, str]:
        """Process image using OpenAI's vision API. Raises an exception on connection errors 
           so that the fallback logic is triggered."""
        processor = OpenaiVisionProcessor(self.openai_client)
        result = processor.process_image_with_vision(image_path, query, input_data)

        if result.get("status") == "error" and "connection error" in result.get("error", "").lower():
            import httpx
            raise httpx.ConnectError(f"OpenAI Connection Error: {result['error']}")

        return result

    async def _anthropic_process_image_with_vision(self, image_path: str, query: str, input_data: List[FileDataInfo]) -> Dict[str, str]:
        """Process image using Anthropic's vision API"""
        processor = AnthropicVisionProcessor(self.anthropic_client)
        return processor.process_image_with_vision(image_path, query, input_data)

    async def _openai_process_pdf_with_vision(
        self, 
        pdf_path: str, 
        query: str, 
        input_data: List[FileDataInfo], 
        page_range: Optional[tuple[int, int]] = None
    ) -> Dict[str, str]:
        """Process PDF using OpenAI's vision API"""
        processor = OpenaiVisionProcessor(self.openai_client)
        # Ensure page_range values are integers if provided
        if page_range:
            page_range = (int(page_range[0]), int(page_range[1]))
        result = processor.process_pdf_with_vision(pdf_path, query, input_data, page_range)

        # If the returned dict indicates an error related to connection, raise an actual exception
        if result.get("status") == "error" and "connection error" in result.get("error", "").lower():
            import httpx
            # Raise a ConnectError so execute_with_fallback will catch it
            raise httpx.ConnectError(f"OpenAI Connection Error: {result['error']}")

        return result

    async def _anthropic_process_pdf_with_vision(
        self, 
        pdf_path: str, 
        query: str, 
        input_data: List[FileDataInfo], 
        page_range: Optional[tuple[int, int]] = None
    ) -> Dict[str, str]:
        """Process PDF using Anthropic's vision API"""
        processor = AnthropicVisionProcessor(self.anthropic_client)
        # Ensure page_range values are integers if provided
        if page_range:
            page_range = (int(page_range[0]), int(page_range[1]))
        return processor.process_pdf_with_vision(pdf_path, query, input_data, page_range)

    
    
    async def _openai_gen_from_query(
        self, 
        query: str, 
        data: List[FileDataInfo],
        batch_context: Optional[Dict[str, int]] = None
    ) -> str:
        data_description = self._build_data_description(data)
        
        # Add batch context to prompt if available
        batch_info = ""
        if batch_context:
            batch_info = f"\nUser files are being processed in batches.  This is batch {batch_context['current']} of {batch_context['total']}"
        
        user_content = f"Available Data:\n{data_description}\n\nQuery:\n{query}{batch_info}"
        
        response = await self._openai_generate_text(
            system_prompt=self._gen_from_query_prompt,
            user_content=user_content
        )
        print(f"\n -------  gen_from_query called with user content:  ------- \n {user_content}")
        return response

    async def _anthropic_gen_from_query(
        self, 
        query: str, 
        data: List[FileDataInfo],
        batch_context: Optional[Dict[str, int]] = None
    ) -> str:
        data_description = self._build_data_description(data)
        
        # Add batch context to prompt if available
        batch_info = ""
        if batch_context:
            batch_info = f"\nUser files are being processed in batches.  This is batch {batch_context['current']} of {batch_context['total']}"
        
        user_content = f"Available Data:\n{data_description}\n\nQuery:\n{query}{batch_info}"
        
        response = await self._anthropic_generate_text(
            system_prompt=self._gen_from_query_prompt,
            user_content=user_content
        )
        return response

    async def _openai_gen_from_error(self, result: SandboxResult, error_attempts: int, 
                                   data: List[FileDataInfo], past_errors: List[str]) -> str:
        data_description = self._build_data_description(data)
        user_content = f"""Here is the original available data, user query, code, past errors, and new error
        - try not to repeat any of the past errors in your new solution:
        Available Data:\n{data_description}\n\n
        Original Query:\n{result.original_query}\n\n
        Code:\n{result.code}\n\n
        Past Errors:\n{past_errors}\n\n
        New Error:\n{result.error}
        """
        
        response = await self._openai_generate_text(
            system_prompt=self._gen_from_error_prompt,
            user_content=user_content
        )
        print(f"""Gen from error called, attempt: {error_attempts}, query: \n{result.original_query} 
              \ncode: \n{result.code} \nerror: \n{result.error}""")
        return response

    async def _anthropic_gen_from_error(self, result: SandboxResult, error_attempts: int,
                                      data: List[FileDataInfo], past_errors: List[str]) -> str:
        data_description = self._build_data_description(data)
        user_content = f"""Here is the original available data, user query, code, past errors, and new error
                - try not to repeat any of the past errors in your new solution:
                Available Data:\n{data_description}\n\n
                Original Query:\n{result.original_query}\n\n
                Code:\n{result.code}\n\n
                Past Errors:\n{past_errors}\n\n
                New Error:\n{result.error}"""
        
        response = await self._anthropic_generate_text(
            system_prompt=self._gen_from_error_prompt,
            user_content=user_content
        )
        print(f"""Gen from error called, attempt: {error_attempts}, query: \n{result.original_query} 
              \ncode: \n{result.code} \nerror: \n{result.error}""")
        return response

    async def _openai_gen_from_analysis(self, result: SandboxResult, analysis_result: str,
                                      data: List[FileDataInfo], past_errors: List[str]) -> str:
        data_description = self._build_data_description(data)
        user_content = f"""Original Query:\n{result.original_query}\n
        Available Data:\n{data_description}\n
        Code:\n{result.code}\n
        Analysis:\n{analysis_result}\n
        Past Errors:\n{past_errors}
        """
        
        return await self._openai_generate_text(
            system_prompt=self._gen_from_analysis_prompt,
            user_content=user_content
        )

    async def _anthropic_gen_from_analysis(self, result: SandboxResult, analysis_result: str,
                                         data: List[FileDataInfo], past_errors: List[str]) -> str:
        data_description = self._build_data_description(data)
        user_content = f"""Original Query:\n{result.original_query}\n
                Available Data:\n{data_description}\n
                Code:\n{result.code}\n
                Analysis:\n{analysis_result}\n
                Past Errors:\n{past_errors}
        """
        
        return await self._anthropic_generate_text(
            system_prompt=self._gen_from_analysis_prompt,
            user_content=user_content
        )

    async def _openai_analyze_sandbox_result(
        self, 
        result: SandboxResult, 
        old_data: List[FileDataInfo],
        new_data: FileDataInfo, 
        analyzer_context: str,
        batch_context: Optional[Dict[str, int]] = None
    ) -> str:
        old_data_snapshot = self._build_old_data_snapshot(old_data)
        if len(analyzer_context) < 10:
            analyzer_context = "No dataset diff information provided"

        # Add batch context to prompt if available
        batch_info = ""
        if batch_context:
            batch_info = f"\nUser files are being processed in batches. This is batch {batch_context['current']} of {batch_context['total']}"
        
        user_content = f""" 
    Here is the original user query, snapshots of input data, error free code, a snapshot of the result, and dataset diff information:
    Original Query:\n{result.original_query}\n
    Input Data Snapshots:\n{old_data_snapshot}\n
    Result Snapshot:\n{new_data.snapshot}\n
    {batch_info}\n
    Dataset Diff Information:\n{analyzer_context}\n
    """

        response = await self._openai_generate_text(
            system_prompt=self._analyze_sandbox_prompt,
            user_content=user_content
        )
        print(f"\n ------- analyze_sandbox_result called with user content: ------- \n {user_content}")
        return response

    async def _anthropic_analyze_sandbox_result(
        self, 
        result: SandboxResult, 
        old_data: List[FileDataInfo],
        new_data: FileDataInfo, 
        analyzer_context: str,
        batch_context: Optional[Dict[str, int]] = None
    ) -> str:
        old_data_snapshot = self._build_old_data_snapshot(old_data)
        if len(analyzer_context) < 10:
            analyzer_context = "No dataset diff information provided"

        # Add batch context to prompt if available
        batch_info = ""
        if batch_context:
            batch_info = f"\nUser files are being processed in batches. This is batch {batch_context['current']} of {batch_context['total']}"
        
        user_content = f""" 
    Here is the original user query, snapshots of input data, error free code, a snapshot of the result, and dataset diff information:
    Original Query:\n{result.original_query}\n
    Input Data Snapshots:\n{old_data_snapshot}\n
    Result Snapshot:\n{new_data.snapshot}\n
    {batch_info}\n
    Dataset Diff Information:\n{analyzer_context}\n
    """

        response = await self._anthropic_generate_text(
            system_prompt=self._analyze_sandbox_prompt,
            user_content=user_content
        )
        return response

    async def _openai_sentiment_analysis(self, analysis_result: str) -> Tuple[bool, str]:
        response = await self._openai_generate_text(
            system_prompt=self._sentiment_analysis_prompt,
            user_content=f"Analysis:\n{analysis_result}"
        )
        try:
            result = json.loads(response)
            return result["is_positive"], analysis_result
        except json.JSONDecodeError:
            guess = "true" in response.lower()
            return guess, analysis_result

    async def _anthropic_sentiment_analysis(self, analysis_result: str) -> Tuple[bool, str]:
        response = await self._anthropic_generate_text(
            system_prompt=self._sentiment_analysis_prompt,
            user_content=f"Analysis:\n{analysis_result}"
        )
        try:
            result = json.loads(response)
            return result["is_positive"], analysis_result
        except json.JSONDecodeError:
            guess = "true" in response.lower()
            return guess, analysis_result

    async def _openai_file_namer(self, query: str, data: List[FileDataInfo]) -> str:
        """Generate a filename using OpenAI"""
        data_description = self._build_data_description(data)
        user_content = f"""Based on the query and data below, suggest a filename. 
        Avoid technical language (i.e. dataframe, list, etc.)
        Query: {query}
        Available Data: {data_description}"""
        
        response = self.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_SMALL_MODEL"),
            messages=[
                {"role": "system", "content": self._file_namer_prompt},
                {"role": "user", "content": user_content}
            ]
        ).choices[0].message.content
        time.sleep(float(os.getenv("SLEEP_TIME")))

        
        return self._clean_filename(response)

    async def _anthropic_file_namer(self, query: str, data: List[FileDataInfo]) -> str:
        """Generate a filename using Anthropic"""
        data_description = self._build_data_description(data)
        user_content = f"""Based on the query and data below, suggest a filename. 
        Avoid technical language (i.e. dataframe, list, etc.)
        Query: {query}
        Available Data: {data_description}"""
        
        # Updated to use claude-3-5-haiku-20241022
        response = self.anthropic_client.messages.create(
            model=os.getenv("ANTHROPIC_SMALL_MODEL"),
            max_tokens=5000,
            temperature=0,
            system=self._file_namer_prompt,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": user_content
                        }
                    ]
                }
            ]
        ).content[0].text
        time.sleep(float(os.getenv("SLEEP_TIME")))
        return self._clean_filename(response)

    async def _openai_gen_visualization(
        self,
        data_snapshot: str,
        color_palette: str,
        custom_instructions: Optional[str],
        past_errors: List[str]
    ) -> str:
        """Generate visualization code using OpenAI."""
        print(f"Openai called with past errors: {past_errors}")
        return await self._openai_generate_text(
            system_prompt=self._gen_visualization_prompt,
            user_content=self._build_gen_vis_user_content(data_snapshot, color_palette, custom_instructions, past_errors)
        )

    async def _anthropic_gen_visualization(
        self,
        data_snapshot: str,
        color_palette: str,
        custom_instructions: Optional[str],
        past_errors: List[str]
    ) -> str:
        """Generate visualization code using Anthropic."""


        return await self._anthropic_generate_text(
            system_prompt=self._gen_visualization_prompt,
            user_content=self._build_gen_vis_user_content(data_snapshot, color_palette, custom_instructions, past_errors)
        )

    
    def _build_data_description(self, data: List[FileDataInfo]) -> str:
        if not data:
            return ""
        data_description = ""
        for idx, file_data in enumerate(data):
            var_name = f'data_{idx}' if idx > 0 else 'data'
            data_description += f"\nVariable Name: {var_name}\nData Type: {file_data.data_type}\nSnapshot:\n{file_data.snapshot}\n"
            if hasattr(file_data, 'original_file_name') and file_data.original_file_name:
                data_description += f"Original file name: {file_data.original_file_name}\n"
        return data_description

    def _build_old_data_snapshot(self, old_data: List[FileDataInfo]) -> str:
        old_data_snapshot = ""
        for data in old_data:
            if isinstance(data.snapshot, str):
                data_snapshot = data.snapshot[:MAX_SNAPSHOT_LENGTH] + "...cont'd"
            else:
                data_snapshot = data.snapshot
            old_data_snapshot += f"Original file name: {data.original_file_name}\nData type: {data.data_type}\nData Snapshot:\n{data_snapshot}\n\n"
        return old_data_snapshot

   
    def _build_gen_vis_user_content(self, data_snapshot: str, color_palette: str, custom_instructions: Optional[str], past_errors: List[str]) -> str:
        return f"""Data Snapshot:
        {data_snapshot}
        
        Incorporate these colors into your visualization: {color_palette}
        
        Custom Instructions: {custom_instructions if custom_instructions else 'None provided'}
        
        Generate visualization code following the requirements.
        
        Past Errors(please don't repeat any of these errors in your code): {past_errors}
        """

   
   
    def _clean_filename(self, filename: str) -> str:
        """Clean and standardize filename"""
        filename = filename.strip().lower()
        return "".join(c for c in filename if c.isalnum() or c in ['_', '-'])








