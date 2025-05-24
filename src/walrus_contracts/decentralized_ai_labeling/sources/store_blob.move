module my_contract::blob_storage {
    use sui::object::{UID};
    use sui::tx_context::TxContext;

    struct BlobRecord has key {
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
