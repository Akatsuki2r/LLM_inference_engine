use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::io::{self, Cursor, Read};
use std::path::Path;
use std::fs::File;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid magic bytes")]
    InvalidMagic,
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("Invalid metadata value type ID: {0}")]
    InvalidMetadataValue(u32),
    #[error("Invalid tensor type ID: {0}")]
    InvalidTensorType(u32),
    #[error("UTF-8 decoding error: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
}

impl TryFrom<u32> for GgufValueType {
    type Error = GgufError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GgufValueType::Uint8),
            1 => Ok(GgufValueType::Int8),
            2 => Ok(GgufValueType::Uint16),
            3 => Ok(GgufValueType::Int16),
            4 => Ok(GgufValueType::Uint32),
            5 => Ok(GgufValueType::Int32),
            6 => Ok(GgufValueType::Float32),
            7 => Ok(GgufValueType::Bool),
            8 => Ok(GgufValueType::String),
            _ => Err(GgufError::InvalidMetadataValue(value)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    // Add other types as they are implemented in the kernels
}

impl TryFrom<u32> for GgufTensorType {
    type Error = GgufError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GgufTensorType::F32),
            1 => Ok(GgufTensorType::F16),
            2 => Ok(GgufTensorType::Q4_0),
            _ => Err(GgufError::InvalidTensorType(value)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgufTensorType,
    pub offset: u64,
}

pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u32,
    pub metadata_count: u32,
}

pub struct GgufModel {
    pub header: GgufHeader,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
}

/// Parses a GGUF model from a byte slice.
pub fn parse(data: &[u8]) -> Result<GgufModel, GgufError> {
    let mut cursor = Cursor::new(data);

    // Magic
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(GgufError::InvalidMagic);
    }

    let version = cursor.read_u32::<LittleEndian>()?;
    if version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = cursor.read_u32::<LittleEndian>()?;
    let metadata_count = cursor.read_u32::<LittleEndian>()?;

    let mut metadata = HashMap::new();
    for _ in 0..metadata_count {
        let key = read_string(&mut cursor)?;
        let value_type_id = cursor.read_u32::<LittleEndian>()?;
        let value = read_value(&mut cursor, value_type_id)?;
        metadata.insert(key, value);
    }

    let mut tensors = HashMap::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut cursor)?;
        let n_dims = cursor.read_u32::<LittleEndian>()?;
        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dimensions.push(cursor.read_u64::<LittleEndian>()?);
        }
        let tensor_type_id = cursor.read_u32::<LittleEndian>()?;
        let tensor_type = GgufTensorType::try_from(tensor_type_id)?;
        let offset = cursor.read_u64::<LittleEndian>()?;

        tensors.insert(name.clone(), GgufTensorInfo {
            name,
            dimensions,
            tensor_type,
            offset,
        });
    }

    Ok(GgufModel {
        header: GgufHeader {
            version,
            tensor_count,
            metadata_count,
        },
        metadata,
        tensors,
    })
}

/// Parses a GGUF model from a file by memory-mapping it.
pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<GgufModel, GgufError> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    parse(&mmap)
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String, GgufError> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_value(cursor: &mut Cursor<&[u8]>, value_type_id: u32) -> Result<GgufValue, GgufError> {
    match value_type_id {
        0 => Ok(GgufValue::Uint8(cursor.read_u8()?)),
        1 => Ok(GgufValue::Int8(cursor.read_i8()?)),
        2 => Ok(GgufValue::Uint16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(GgufValue::Int16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(GgufValue::Uint32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(GgufValue::Int32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(GgufValue::Float32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(GgufValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(GgufValue::String(read_string(cursor)?)),
        _ => Err(GgufError::InvalidMetadataValue(value_type_id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_header() {
        // Magic "GGUF", Version 3, Tensor Count 0, Metadata Count 0
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        let model = parse(&data).unwrap();
        assert_eq!(model.header.version, 3);
        assert_eq!(model.header.tensor_count, 0);
        assert_eq!(model.header.metadata_count, 0);
    }

    #[test]
    fn test_parse_metadata() {
        // Magic "GGUF", Version 3, Tensor Count 0, Metadata Count 1
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());

        // Key: "test", type: Uint32, value: 42
        let key = "test";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&4u32.to_le_bytes()); // Uint32
        data.extend_from_slice(&42u32.to_le_bytes());

        let model = parse(&data).unwrap();
        assert_eq!(model.metadata.get("test"), Some(&GgufValue::Uint32(42)));
    }

    #[test]
    fn test_invalid_magic() {
        let data = b"NOTG";
        let result = parse(data);
        assert!(matches!(result, Err(GgufError::InvalidMagic)));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&2u32.to_le_bytes()); // Version 2
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        let result = parse(&data.as_slice());
        assert!(matches!(result, Err(GgufError::UnsupportedVersion(2))));
    }
}
