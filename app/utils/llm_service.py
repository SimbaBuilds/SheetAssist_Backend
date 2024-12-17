from fastapi import HTTPException
from typing import Tuple, Any, Dict, List
from functools import lru_cache
from openai import OpenAI
from anthropic import Anthropic
import logging
from app.schemas import SandboxResult, FileDataInfo
import json
from app.utils.vision_processing import OpenaiVisionProcessor, AnthropicVisionProcessor
from app.utils.system_prompts import gen_from_query_prompt, gen_from_error_prompt, gen_from_analysis_prompt, analyze_sandbox_prompt, sentiment_analysis_prompt, file_namer_prompt

class LLMService:
    def __init__(self):
        self.openai_client = OpenAI()
        self.anthropic_client = Anthropic()
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
            }
        }

        # Add system prompts as class attributes
        self._gen_from_query_prompt = gen_from_query_prompt
        self._gen_from_error_prompt = gen_from_error_prompt
        self._gen_from_analysis_prompt = gen_from_analysis_prompt
        self._analyze_sandbox_prompt = analyze_sandbox_prompt
        self._sentiment_analysis_prompt = sentiment_analysis_prompt
        self._file_namer_prompt = file_namer_prompt


    async def execute_with_fallback(self, operation: str, *args, **kwargs) -> Tuple[str, Any]:
        try:
            # Try OpenAI first
            result = await self._execute_openai(operation, *args, **kwargs)
            return "openai", result
        except Exception as e:
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
            model="gpt-4o",
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
            model="claude-3-5-sonnet-20241022",
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

    async def _openai_process_image_with_vision(self, image_path: str, query: str) -> Dict[str, str]:
        """Process image using OpenAI's vision API"""
        processor = OpenaiVisionProcessor()
        return processor.process_image_with_vision(image_path, query)

    async def _anthropic_process_image_with_vision(self, image_path: str, query: str) -> Dict[str, str]:
        """Process image using Anthropic's vision API"""
        processor = AnthropicVisionProcessor()
        return processor.process_image_with_vision(image_path, query)

    async def _openai_process_pdf_with_vision(self, pdf_path: str, query: str) -> Dict[str, str]:
        """Process image using OpenAI's vision API"""
        processor = OpenaiVisionProcessor()
        return processor.process_pdf_with_vision(pdf_path, query)

    async def _anthropic_process_pdf_with_vision(self, pdf_path: str, query: str) -> Dict[str, str]:
        """Process image using Anthropic's vision API"""
        processor = AnthropicVisionProcessor()
        return processor.process_pdf_with_vision(pdf_path, query)

    
    
    async def _openai_gen_from_query(self, query: str, data: List[FileDataInfo]) -> str:
        data_description = self._build_data_description(data)
        user_content = f"Available Data:\n{data_description}\n\nQuery:\n{query}"
        
        response = await self._openai_generate_text(
            system_prompt=self._gen_from_query_prompt,
            user_content=user_content
        )
        print(f"LLM called with available data: {data_description} and query: {query} \nCode generated from query: \n {response}")
        return response

    async def _anthropic_gen_from_query(self, query: str, data: List[FileDataInfo]) -> str:
        data_description = self._build_data_description(data)
        user_content = f"Available Data:\n{data_description}\n\nQuery:\n{query}"
        
        response = await self._anthropic_generate_text(
            system_prompt=self._gen_from_query_prompt,
            user_content=user_content
        )
        print(f"LLM called with available data: {data_description} and query: {query} \nCode generated from query: \n {response}")
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
                New Error:\n{result.error}"""
        
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
                Past Errors:\n{past_errors}"""
        
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
                Past Errors:\n{past_errors}"""
        
        return await self._anthropic_generate_text(
            system_prompt=self._gen_from_analysis_prompt,
            user_content=user_content
        )

    async def _openai_analyze_sandbox_result(self, result: SandboxResult, old_data: List[FileDataInfo],
                                           new_data: FileDataInfo, analyzer_context: str) -> str:
        old_data_snapshot = self._build_old_data_snapshot(old_data)
        user_content = f""" 
                Here is the original user query, snapshots of old data, error free code, a snapshot of the result, and dataset diff information:
                Original Query:\n{result.original_query}\n
                Old Data Snapshots:\n{old_data_snapshot}\n
                Error Free Code:\n{result.code}\n
                Result Snapshot:\n{new_data.snapshot}\n
                Dataset Diff Information:\n{analyzer_context}\n
                """

        response = await self._openai_generate_text(
            system_prompt=self._analyze_sandbox_prompt,
            user_content=user_content
        )
        print(f"""\n\nSandbox result analyzer called with Query: {result.original_query}\n
                Old Data Snapshots:\n{old_data_snapshot}\n
                Error Free Code:\n{result.code}\n
                Result Snapshot:\n{new_data.snapshot}\n
                Dataset Diff Information:\n{analyzer_context}\n\n
                """
        )
        return response

    async def _anthropic_analyze_sandbox_result(self, result: SandboxResult, old_data: List[FileDataInfo],
                                              new_data: FileDataInfo, analyzer_context: str) -> str:
        old_data_snapshot = self._build_old_data_snapshot(old_data)
        user_content = f""" 
                Here is the original user query, snapshots of old data, error free code, a snapshot of the result, and dataset diff information:
                Original Query:\n{result.original_query}\n
                Old Data Snapshots:\n{old_data_snapshot}\n
                Error Free Code:\n{result.code}\n
                Result Snapshot:\n{new_data.snapshot}\n
                Dataset Diff Information:\n{analyzer_context}\n
                """

        return await self._anthropic_generate_text(
            system_prompt=self._analyze_sandbox_prompt,
            user_content=user_content
        )

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
        
        response = await self._openai_generate_text(
            system_prompt=self._file_namer_prompt,
            user_content=user_content
        )
        return self._clean_filename(response)

    async def _anthropic_file_namer(self, query: str, data: List[FileDataInfo]) -> str:
        """Generate a filename using Anthropic"""
        data_description = self._build_data_description(data)
        user_content = f"""Based on the query and data below, suggest a filename. 
                    Avoid technical language (i.e. dataframe, list, etc.)
                    Query: {query}
                    Available Data: {data_description}"""
        
        response = await self._anthropic_generate_text(
            system_prompt=self._file_namer_prompt,
            user_content=user_content
        )
        return self._clean_filename(response)


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
                data_snapshot = data.snapshot[:500] + "...cont'd"
            else:
                data_snapshot = data.snapshot
            old_data_snapshot += f"Original file name: {data.original_file_name}\nData type: {data.data_type}\nData Snapshot:\n{data_snapshot}\n\n"
        return old_data_snapshot

    def _clean_filename(self, filename: str) -> str:
        """Clean and standardize filename"""
        filename = filename.strip().lower()
        return "".join(c for c in filename if c.isalnum() or c in ['_', '-'])

@lru_cache()
def get_llm_service() -> LLMService:
    return LLMService()