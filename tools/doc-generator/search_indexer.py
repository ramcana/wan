"""
Documentation Search Indexer

Advanced search indexing for WAN22 documentation with full-text search,
tag-based filtering, and content categorization.
"""

import os
import re
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
import sqlite3
from datetime import datetime
import hashlib


@dataclass
class SearchDocument:
    """Document for search indexing"""
    id: str
    title: str
    path: str
    url: str
    content: str
    category: str
    tags: List[str]
    last_updated: str
    word_count: int
    headings: List[str]
    code_blocks: List[str]


@dataclass
class SearchResult:
    """Search result with relevance scoring"""
    document: SearchDocument
    score: float
    matches: List[Dict[str, Any]]
    snippet: str


class SearchIndexer:
    """
    Advanced search indexer for documentation
    """
    
    def __init__(self, docs_root: Path, index_db_path: Optional[Path] = None):
        self.docs_root = Path(docs_root)
        self.index_db_path = index_db_path or (self.docs_root / '.search' / 'index.db')
        self.index_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Stop words for search
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'you', 'your', 'this', 'these',
            'they', 'them', 'their', 'have', 'had', 'can', 'could', 'should',
            'would', 'may', 'might', 'must', 'shall', 'do', 'does', 'did'
        }
    
    def _init_database(self):
        """Initialize SQLite database for search index"""
        with sqlite3.connect(self.index_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL,
                    url TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    headings TEXT NOT NULL,
                    code_blocks TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS search_terms (
                    term TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    positions TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_search_terms_term 
                ON search_terms (term)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_search_terms_document 
                ON search_terms (document_id)
            ''')
            
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    title, content, category, tags, headings,
                    content='documents',
                    content_rowid='rowid'
                )
            ''')
    
    def index_documentation(self) -> Dict[str, Any]:
        """
        Index all documentation files
        """
        indexed_count = 0
        updated_count = 0
        error_count = 0
        errors = []
        
        print("Indexing documentation...")
        
        # Find all markdown files
        for md_file in self.docs_root.rglob('*.md'):
            if self._should_skip_file(md_file):
                continue
            
            try:
                # Check if file needs reindexing
                if self._needs_reindexing(md_file):
                    document = self._process_document(md_file)
                    if document:
                        self._store_document(document)
                        if self._document_exists(document.id):
                            updated_count += 1
                        else:
                            indexed_count += 1
                
            except Exception as e:
                error_count += 1
                errors.append(f"{md_file}: {e}")
                print(f"Error indexing {md_file}: {e}")
        
        # Update FTS index
        self._update_fts_index()
        
        result = {
            'indexed': indexed_count,
            'updated': updated_count,
            'errors': error_count,
            'error_details': errors,
            'total_documents': self._get_document_count()
        }
        
        print(f"Indexing complete: {indexed_count} new, {updated_count} updated, {error_count} errors")
        return result
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            'templates',
            '.metadata',
            '.search',
            'node_modules',
            '.git'
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _needs_reindexing(self, file_path: Path) -> bool:
        """Check if file needs to be reindexed"""
        try:
            # Calculate current file hash
            with open(file_path, 'rb') as f:
                current_hash = hashlib.md5(f.read()).hexdigest()
            
            # Check stored hash
            relative_path = str(file_path.relative_to(self.docs_root))
            doc_id = self._generate_document_id(relative_path)
            
            with sqlite3.connect(self.index_db_path) as conn:
                cursor = conn.execute(
                    'SELECT content_hash FROM documents WHERE id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                
                if result is None:
                    return True  # New document
                
                stored_hash = result[0]
                return current_hash != stored_hash
                
        except Exception:
            return True  # Reindex on error
    
    def _process_document(self, file_path: Path) -> Optional[SearchDocument]:
        """Process a documentation file for indexing"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata
            metadata = self._extract_metadata(content)
            
            # Extract content components
            title = metadata.get('title', self._extract_title_from_content(content))
            clean_content = self._clean_content_for_indexing(content)
            headings = self._extract_headings(content)
            code_blocks = self._extract_code_blocks(content)
            
            # Generate document info
            relative_path = str(file_path.relative_to(self.docs_root))
            doc_id = self._generate_document_id(relative_path)
            url = relative_path.replace('.md', '.html')
            
            # Calculate content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            return SearchDocument(
                id=doc_id,
                title=title,
                path=relative_path,
                url=url,
                content=clean_content,
                category=metadata.get('category', 'reference'),
                tags=metadata.get('tags', []),
                last_updated=metadata.get('last_updated', datetime.now().strftime('%Y-%m-%d')),
                word_count=len(clean_content.split()),
                headings=headings,
                code_blocks=code_blocks
            )
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter metadata"""
        if not content.startswith('---'):
            return {}
        
        try:
            end_marker = content.find('---', 3)
            if end_marker == -1:
                return {}
            
            frontmatter = content[3:end_marker].strip()
            metadata = yaml.safe_load(frontmatter) or {}
            
            # Ensure tags is a list
            if 'tags' in metadata:
                if isinstance(metadata['tags'], str):
                    metadata['tags'] = [tag.strip() for tag in metadata['tags'].split(',')]
            
            return metadata
            
        except Exception:
            return {}
    
    def _extract_title_from_content(self, content: str) -> str:
        """Extract title from first heading"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return "Untitled"
    
    def _clean_content_for_indexing(self, content: str) -> str:
        """Clean content for search indexing"""
        # Remove frontmatter
        if content.startswith('---'):
            end_marker = content.find('---', 3)
            if end_marker != -1:
                content = content[end_marker + 3:]
        
        # Remove markdown formatting
        content = re.sub(r'```[\s\S]*?```', '', content)  # Code blocks
        content = re.sub(r'`[^`]+`', '', content)  # Inline code
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Images
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # Links
        content = re.sub(r'[#*_~`]', '', content)  # Markdown symbols
        content = re.sub(r'\n+', ' ', content)  # Multiple newlines
        content = re.sub(r'\s+', ' ', content)  # Multiple spaces
        
        return content.strip()
    
    def _extract_headings(self, content: str) -> List[str]:
        """Extract all headings from content"""
        headings = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # Extract heading text
                heading_text = re.sub(r'^#+\s*', '', line).strip()
                if heading_text:
                    headings.append(heading_text)
        
        return headings
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from content"""
        code_blocks = []
        
        # Find fenced code blocks
        pattern = r'```[\w]*\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        code_blocks.extend(matches)
        
        # Find inline code
        inline_pattern = r'`([^`]+)`'
        inline_matches = re.findall(inline_pattern, content)
        code_blocks.extend(inline_matches)
        
        return code_blocks
    
    def _generate_document_id(self, relative_path: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(relative_path.encode()).hexdigest()
    
    def _store_document(self, document: SearchDocument):
        """Store document in search index"""
        with sqlite3.connect(self.index_db_path) as conn:
            # Store document
            conn.execute('''
                INSERT OR REPLACE INTO documents 
                (id, title, path, url, content, category, tags, last_updated, 
                 word_count, headings, code_blocks, content_hash, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document.id,
                document.title,
                document.path,
                document.url,
                document.content,
                document.category,
                json.dumps(document.tags),
                document.last_updated,
                document.word_count,
                json.dumps(document.headings),
                json.dumps(document.code_blocks),
                hashlib.md5(document.content.encode()).hexdigest(),
                datetime.now().isoformat()
            ))
            
            # Store search terms
            self._index_document_terms(conn, document)
    
    def _index_document_terms(self, conn: sqlite3.Connection, document: SearchDocument):
        """Index individual terms for a document"""
        # Delete existing terms for this document
        conn.execute('DELETE FROM search_terms WHERE document_id = ?', (document.id,))
        
        # Tokenize content
        terms = self._tokenize_content(document.content + ' ' + document.title)
        
        # Count term frequencies and positions
        term_data = {}
        for position, term in enumerate(terms):
            if term not in self.stop_words and len(term) > 2:
                if term not in term_data:
                    term_data[term] = {'frequency': 0, 'positions': []}
                term_data[term]['frequency'] += 1
                term_data[term]['positions'].append(position)
        
        # Store terms
        for term, data in term_data.items():
            conn.execute('''
                INSERT INTO search_terms (term, document_id, frequency, positions)
                VALUES (?, ?, ?, ?)
            ''', (
                term,
                document.id,
                data['frequency'],
                json.dumps(data['positions'])
            ))
    
    def _tokenize_content(self, content: str) -> List[str]:
        """Tokenize content into searchable terms"""
        # Convert to lowercase and split on non-alphanumeric characters
        terms = re.findall(r'\b[a-zA-Z0-9]+\b', content.lower())
        return terms
    
    def _document_exists(self, doc_id: str) -> bool:
        """Check if document already exists in index"""
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.execute('SELECT 1 FROM documents WHERE id = ?', (doc_id,))
            return cursor.fetchone() is not None
    
    def _get_document_count(self) -> int:
        """Get total number of indexed documents"""
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM documents')
            return cursor.fetchone()[0]
    
    def _update_fts_index(self):
        """Update full-text search index"""
        with sqlite3.connect(self.index_db_path) as conn:
            # Rebuild FTS index
            conn.execute('DELETE FROM documents_fts')
            conn.execute('''
                INSERT INTO documents_fts (rowid, title, content, category, tags, headings)
                SELECT rowid, title, content, category, tags, headings FROM documents
            ''')
    
    def search(self, query: str, category: Optional[str] = None, 
               tags: Optional[List[str]] = None, limit: int = 20) -> List[SearchResult]:
        """
        Search documentation with advanced filtering
        """
        if not query.strip():
            return []
        
        results = []
        
        # Use FTS for primary search
        fts_results = self._fts_search(query, category, tags, limit * 2)
        
        # Score and rank results
        for doc_data in fts_results:
            document = self._create_document_from_data(doc_data)
            score = self._calculate_relevance_score(document, query)
            matches = self._find_matches(document, query)
            snippet = self._generate_snippet(document, query)
            
            results.append(SearchResult(
                document=document,
                score=score,
                matches=matches,
                snippet=snippet
            ))
        
        # Sort by relevance score
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results[:limit]
    
    def _fts_search(self, query: str, category: Optional[str], 
                   tags: Optional[List[str]], limit: int) -> List[Dict[str, Any]]:
        """Full-text search using SQLite FTS"""
        with sqlite3.connect(self.index_db_path) as conn:
            # Build FTS query
            fts_query = self._build_fts_query(query)
            
            # Base query
            sql = '''
                SELECT d.* FROM documents d
                JOIN documents_fts fts ON d.rowid = fts.rowid
                WHERE documents_fts MATCH ?
            '''
            params = [fts_query]
            
            # Add category filter
            if category:
                sql += ' AND d.category = ?'
                params.append(category)
            
            # Add tag filter
            if tags:
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append('d.tags LIKE ?')
                    params.append(f'%"{tag}"%')
                
                if tag_conditions:
                    sql += ' AND (' + ' OR '.join(tag_conditions) + ')'
            
            sql += ' ORDER BY bm25(documents_fts) LIMIT ?'
            params.append(limit)
            
            cursor = conn.execute(sql, params)
            columns = [desc[0] for desc in cursor.description]
            
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
    
    def _build_fts_query(self, query: str) -> str:
        """Build FTS query from user input"""
        # Tokenize query
        terms = self._tokenize_content(query)
        
        # Remove stop words
        terms = [term for term in terms if term not in self.stop_words and len(term) > 2]
        
        if not terms:
            return query
        
        # Build FTS query with OR logic
        fts_terms = []
        for term in terms:
            fts_terms.append(f'"{term}"')
        
        return ' OR '.join(fts_terms)
    
    def _create_document_from_data(self, data: Dict[str, Any]) -> SearchDocument:
        """Create SearchDocument from database row"""
        return SearchDocument(
            id=data['id'],
            title=data['title'],
            path=data['path'],
            url=data['url'],
            content=data['content'],
            category=data['category'],
            tags=json.loads(data['tags']),
            last_updated=data['last_updated'],
            word_count=data['word_count'],
            headings=json.loads(data['headings']),
            code_blocks=json.loads(data['code_blocks'])
        )
    
    def _calculate_relevance_score(self, document: SearchDocument, query: str) -> float:
        """Calculate relevance score for search result"""
        score = 0.0
        query_terms = self._tokenize_content(query)
        
        # Title match bonus
        title_terms = self._tokenize_content(document.title)
        title_matches = len(set(query_terms) & set(title_terms))
        score += title_matches * 10.0
        
        # Content match score
        content_terms = self._tokenize_content(document.content)
        content_matches = len(set(query_terms) & set(content_terms))
        score += content_matches * 1.0
        
        # Heading match bonus
        heading_terms = self._tokenize_content(' '.join(document.headings))
        heading_matches = len(set(query_terms) & set(heading_terms))
        score += heading_matches * 5.0
        
        # Category relevance
        if any(term in document.category.lower() for term in query_terms):
            score += 3.0
        
        # Tag relevance
        tag_terms = self._tokenize_content(' '.join(document.tags))
        tag_matches = len(set(query_terms) & set(tag_terms))
        score += tag_matches * 2.0
        
        # Normalize by document length
        if document.word_count > 0:
            score = score / (1 + document.word_count / 1000.0)
        
        return score
    
    def _find_matches(self, document: SearchDocument, query: str) -> List[Dict[str, Any]]:
        """Find specific matches in document"""
        matches = []
        query_terms = self._tokenize_content(query)
        
        # Find matches in title
        for term in query_terms:
            if term in document.title.lower():
                matches.append({
                    'type': 'title',
                    'term': term,
                    'context': document.title
                })
        
        # Find matches in headings
        for heading in document.headings:
            for term in query_terms:
                if term in heading.lower():
                    matches.append({
                        'type': 'heading',
                        'term': term,
                        'context': heading
                    })
        
        return matches
    
    def _generate_snippet(self, document: SearchDocument, query: str, 
                         max_length: int = 200) -> str:
        """Generate search result snippet"""
        query_terms = self._tokenize_content(query)
        content = document.content
        
        # Find best snippet location
        best_position = 0
        best_score = 0
        
        words = content.split()
        for i in range(len(words) - 20):
            snippet_words = words[i:i + 20]
            snippet_text = ' '.join(snippet_words).lower()
            
            # Score this snippet
            score = sum(1 for term in query_terms if term in snippet_text)
            if score > best_score:
                best_score = score
                best_position = i
        
        # Extract snippet
        snippet_words = words[best_position:best_position + 30]
        snippet = ' '.join(snippet_words)
        
        # Truncate if too long
        if len(snippet) > max_length:
            snippet = snippet[:max_length] + '...'
        
        # Highlight query terms
        for term in query_terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            snippet = pattern.sub(f'<mark>{term}</mark>', snippet)
        
        return snippet
    
    def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get search suggestions based on partial query"""
        if len(partial_query) < 2:
            return []
        
        suggestions = []
        
        with sqlite3.connect(self.index_db_path) as conn:
            # Find terms that start with the partial query
            cursor = conn.execute('''
                SELECT DISTINCT term FROM search_terms 
                WHERE term LIKE ? 
                ORDER BY frequency DESC 
                LIMIT ?
            ''', (f'{partial_query}%', limit))
            
            suggestions = [row[0] for row in cursor.fetchall()]
        
        return suggestions
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get search index statistics"""
        with sqlite3.connect(self.index_db_path) as conn:
            # Document count by category
            cursor = conn.execute('''
                SELECT category, COUNT(*) as count 
                FROM documents 
                GROUP BY category
            ''')
            categories = dict(cursor.fetchall())
            
            # Total documents
            cursor = conn.execute('SELECT COUNT(*) FROM documents')
            total_docs = cursor.fetchone()[0]
            
            # Total terms
            cursor = conn.execute('SELECT COUNT(DISTINCT term) FROM search_terms')
            total_terms = cursor.fetchone()[0]
            
            # Average document length
            cursor = conn.execute('SELECT AVG(word_count) FROM documents')
            avg_length = cursor.fetchone()[0] or 0
            
            return {
                'total_documents': total_docs,
                'total_terms': total_terms,
                'average_document_length': round(avg_length, 1),
                'documents_by_category': categories,
                'index_size': os.path.getsize(self.index_db_path) if self.index_db_path.exists() else 0
            }


def main():
    """CLI interface for search indexer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Documentation Search Indexer')
    parser.add_argument('command', choices=['index', 'search', 'stats'], 
                       help='Command to execute')
    parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    parser.add_argument('--query', help='Search query (for search command)')
    parser.add_argument('--category', help='Filter by category')
    parser.add_argument('--tags', nargs='+', help='Filter by tags')
    parser.add_argument('--limit', type=int, default=10, help='Result limit')
    
    args = parser.parse_args()
    
    indexer = SearchIndexer(Path(args.docs_dir))
    
    if args.command == 'index':
        result = indexer.index_documentation()
        print(f"Indexing complete: {result}")
    
    elif args.command == 'search':
        if not args.query:
            print("Error: --query required for search command")
            return
        
        results = indexer.search(args.query, args.category, args.tags, args.limit)
        
        print(f"Found {len(results)} results for '{args.query}':")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.document.title}")
            print(f"   Path: {result.document.path}")
            print(f"   Score: {result.score:.2f}")
            print(f"   Snippet: {result.snippet}")
    
    elif args.command == 'stats':
        stats = indexer.get_index_stats()
        print("Search Index Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()