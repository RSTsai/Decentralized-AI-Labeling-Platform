module decentralized_ai_labeling::blob_storage {

    public struct BlobRecord has key {
        id: UID,
        content: vector<u8>,
        owner: address
    }

    public entry fun store_blob(
        blob: vector<u8>,
        owner: &mut TxContext
    ) {
        let record = BlobRecord {
            id: object::new(owner),
            content: blob,
            owner: tx_context::sender(owner)
        };
        transfer::transfer(record, tx_context::sender(owner));
    }
}
