import { TransactionBlock } from "@mysten/sui.js";

async function storeBlobOnChain() {
  const file = document.getElementById("file-input").files[0];
  const arrayBuffer = await file.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);

  const tx = new TransactionBlock();
  tx.moveCall({
    target: "0xYOUR_PACKAGE_ID::blob_storage::store_blob",
    arguments: [
      tx.pure(bytes),
    ],
  });

  const signedTx = await window.sui.signTransactionBlock({
    transactionBlock: tx,
  });
  
  const result = await window.sui.executeTransactionBlock({
    transactionBlock: signedTx.transactionBlockBytes,
    signature: signedTx.signature,
  });
  
  return result;
}